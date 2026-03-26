#include "lemon/backend_manager.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/fastflowlm_server.h"
#include "lemon/system_info.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <lemon/utils/aixlog.hpp>

namespace fs = std::filesystem;

namespace lemon {

namespace {

static const char* ROCM_STABLE_RUNTIME_DIR = "rocm-stable-runtime";

std::string normalize_backend_name(const std::string& recipe, const std::string& backend) {
    if (recipe == "llamacpp" && backend == "rocm") {
        return "rocm-preview";
    }
    return backend;
}

bool is_rocm_runtime_available() {
#ifdef _WIN32
    return !utils::find_executable_in_path("amdhip64.dll").empty();
#else
    const std::vector<std::string> paths = {
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib64/libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so.7",
        "/opt/rocm/lib64/libamdhip64.so.7",
        "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
        "/usr/lib/x86_64-linux-gnu/libamdhip64.so.7"
    };

    for (const auto& path : paths) {
        if (fs::exists(path)) {
            return true;
        }
    }
    return false;
#endif
}

std::string get_rocm_stable_runtime_version(const json& backend_versions) {
    if (backend_versions.contains("llamacpp") &&
        backend_versions["llamacpp"].is_object() &&
        backend_versions["llamacpp"].contains("rocm-stable-runtime") &&
        backend_versions["llamacpp"]["rocm-stable-runtime"].is_string()) {
        return backend_versions["llamacpp"]["rocm-stable-runtime"].get<std::string>();
    }

    return "v7.2";
}

std::string get_rocm_stable_runtime_asset_filename(const std::string& version) {
#ifdef __linux__
    return "rocm-7.2-runtime-libs.tar.gz";
#else
    throw std::runtime_error("ROCm stable runtime artifacts are currently supported on Linux only");
#endif
}

void install_rocm_stable_runtime_if_needed(const backends::BackendSpec& spec,
                                           const json& backend_versions,
                                           DownloadProgressCallback progress_cb) {
    if (is_rocm_runtime_available()) {
        LOG(DEBUG, "BackendManager") << "System ROCm runtime detected; skipping rocm-stable runtime install" << std::endl;
        return;
    }

    const std::string version = get_rocm_stable_runtime_version(backend_versions);
    const std::string repo = "lemonade-sdk/rocm-stable";
    const std::string filename = get_rocm_stable_runtime_asset_filename(version);
    const std::string install_dir = backends::BackendUtils::get_install_directory(ROCM_STABLE_RUNTIME_DIR, "");
    const fs::path version_file = fs::path(install_dir) / "version.txt";

    if (fs::exists(version_file)) {
        std::ifstream vf(version_file);
        std::string installed_version;
        std::getline(vf, installed_version);
        if (installed_version == version) {
            return;
        }
    }

    fs::remove_all(install_dir);
    fs::create_directories(install_dir);

    const std::string url = "https://github.com/" + repo + "/releases/download/" + version + "/" + filename;
    const std::string extension = filename.size() >= 7 && filename.substr(filename.size() - 7) == ".tar.gz"
        ? ".tar.gz"
        : ".zip";
    std::string archive_basename = filename;
    for (char& ch : archive_basename) {
        if (ch == '/' || ch == '\\' || ch == ':') {
            ch = '_';
        }
    }
    const std::string archive_path = (fs::temp_directory_path() /
        ("llamacpp_rocm_stable_runtime_" + version + "_" + archive_basename)).string();

    // Remove any stale archive from a previous failed download. The HTTP downloader
    // supports resume-by-default, which is undesirable here because older attempts
    // may have cached an HTML error page under the same temp filename.
    std::error_code archive_ec;
    fs::remove(archive_path, archive_ec);

    utils::ProgressCallback http_progress_cb;
    if (progress_cb) {
        http_progress_cb = [&progress_cb, &filename](size_t downloaded, size_t total) -> bool {
            DownloadProgress p;
            p.file = filename;
            p.file_index = 1;
            p.total_files = 1;
            p.bytes_downloaded = downloaded;
            p.bytes_total = total;
            p.percent = total > 0 ? static_cast<int>((downloaded * 100) / total) : 0;
            p.complete = false;
            return progress_cb(p);
        };
    } else {
        http_progress_cb = utils::create_throttled_progress_callback();
    }

    auto download_result = utils::HttpClient::download_file(url, archive_path, http_progress_cb);
    if (!download_result.success) {
        throw std::runtime_error("Failed to download ROCm stable runtime from: " + url +
                                 " - " + download_result.error_message);
    }

    if (!backends::BackendUtils::extract_archive(archive_path, install_dir, spec.log_name())) {
        fs::remove(archive_path);
        fs::remove_all(install_dir);
        throw std::runtime_error("Failed to extract ROCm stable runtime archive: " + archive_path);
    }

    {
        std::ofstream vf(version_file);
        vf << version;
    }

    fs::remove(archive_path);

    if (progress_cb) {
        DownloadProgress p;
        p.file = filename;
        p.file_index = 1;
        p.total_files = 1;
        p.bytes_downloaded = download_result.bytes_downloaded;
        p.bytes_total = download_result.total_bytes;
        p.percent = 100;
        p.complete = true;
        progress_cb(p);
    }
}

} // namespace

BackendManager::BackendManager() {
    try {
        std::string config_path = utils::get_resource_path("resources/backend_versions.json");
        backend_versions_ = utils::JsonUtils::load_from_file(config_path);
    } catch (const std::exception& e) {
        LOG(WARNING, "BackendManager") << "Could not load backend_versions.json: " << e.what() << std::endl;
        backend_versions_ = json::object();
    }
}

std::string BackendManager::get_version_from_config(const std::string& recipe, const std::string& backend) {
    std::string resolved_backend = normalize_backend_name(recipe, backend);

    // The "system" backend doesn't have a version in backend_versions.json
    // because it uses a pre-installed binary from the system PATH
    if (resolved_backend == "system") {
        return "";
    }

    if (!backend_versions_.contains(recipe) || !backend_versions_[recipe].is_object()) {
        throw std::runtime_error("backend_versions.json is missing '" + recipe + "' section");
    }
    const auto& recipe_config = backend_versions_[recipe];
    if (!recipe_config.contains(resolved_backend) || !recipe_config[resolved_backend].is_string()) {
        throw std::runtime_error("backend_versions.json is missing version for: " + recipe + ":" + resolved_backend);
    }
    return recipe_config[resolved_backend].get<std::string>();
}

// ============================================================================
// Core operations
// ============================================================================

// ============================================================================
// Install parameters
// ============================================================================

BackendManager::InstallParams BackendManager::get_install_params(const std::string& recipe, const std::string& backend) {
    std::string resolved_backend = normalize_backend_name(recipe, backend);

    if (recipe == "flm") {
        throw std::runtime_error("FLM uses a special installer and cannot be installed via get_install_params");
    }

    auto* spec = backends::try_get_spec_for_recipe(recipe);
    if (!spec) {
        throw std::runtime_error("[BackendManager] Unknown recipe: " + recipe);
    }
    std::string version = get_version_from_config(recipe, resolved_backend);

    if (!spec->install_params_fn) {
        throw std::runtime_error("No install params function for recipe: " + recipe);
    }

    auto params = spec->install_params_fn(resolved_backend, version);
    return {params.repo, params.filename, version};
}

void BackendManager::install_backend(const std::string& recipe, const std::string& backend,
                                     DownloadProgressCallback progress_cb) {
    std::string resolved_backend = normalize_backend_name(recipe, backend);
    LOG(DEBUG, "BackendManager") << "Installing " << recipe << ":" << resolved_backend << std::endl;

    // System backend uses a pre-installed binary from PATH - nothing to install
    if (resolved_backend == "system") {
        return;
    }

    // FLM special case - uses installer exe with its own install logic
    if (recipe == "flm") {
        auto status = SystemInfoCache::get_flm_status();
        if (status.state == "installed") {
            // Already installed — nothing to do
        } else if (status.state == "unsupported") {
            throw std::runtime_error("FLM is not supported on this system: " + status.message);
        } else {
            // installable, update_required, or action_required
            backends::FastFlowLMServer flm_installer("info", nullptr, this);
            flm_installer.install(backend);
            // install() calls SystemInfoCache::invalidate_recipes()
        }
        // Re-read status after install
        status = SystemInfoCache::get_flm_status();
        if (!status.is_ready()) {
            throw std::runtime_error("FLM installation incomplete: " + status.message +
                (status.action.empty() ? "" : ". " + status.action));
        }
        return;
    }

    auto params = get_install_params(recipe, resolved_backend);
    auto* spec = backends::try_get_spec_for_recipe(recipe);
    if (!spec) {
        throw std::runtime_error("[BackendManager] Unknown recipe: " + recipe);
    }

    backends::BackendUtils::install_from_github(
        *spec, params.version, params.repo, params.filename, resolved_backend, progress_cb);

    if (recipe == "llamacpp" && resolved_backend == "rocm-stable") {
        install_rocm_stable_runtime_if_needed(*spec, backend_versions_, progress_cb);
    }
}

void BackendManager::uninstall_backend(const std::string& recipe, const std::string& backend) {
    std::string resolved_backend = normalize_backend_name(recipe, backend);
    LOG(DEBUG, "BackendManager") << "Uninstalling " << recipe << ":" << resolved_backend << std::endl;

    if (recipe == "flm") {
        throw std::runtime_error("Uninstall FastFlowLM using their Windows uninstaller.");
    }

    auto* spec = backends::try_get_spec_for_recipe(recipe);
    if (!spec) {
        throw std::runtime_error("[BackendManager] Unknown recipe: " + recipe);
    }

    std::string install_dir = backends::BackendUtils::get_install_directory(spec->recipe, resolved_backend);

    if (fs::exists(install_dir)) {
        // On Windows, antivirus scanning or indexing can briefly lock files after extraction.
        // Retry a few times with a short delay to handle transient locks.
        std::error_code ec;
        for (int attempt = 0; attempt < 5; ++attempt) {
            fs::remove_all(install_dir, ec);
            if (!ec || !fs::exists(install_dir)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (ec && fs::exists(install_dir)) {
            throw std::runtime_error("Failed to remove " + install_dir + ": " + ec.message());
        }
        LOG(DEBUG, "BackendManager") << "Removed: " << install_dir << std::endl;
    } else {
        LOG(DEBUG, "BackendManager") << "Nothing to uninstall at: " << install_dir << std::endl;
    }

    if (recipe == "llamacpp" && resolved_backend == "rocm-stable") {
        std::string runtime_dir = backends::BackendUtils::get_install_directory(ROCM_STABLE_RUNTIME_DIR, "");
        if (fs::exists(runtime_dir)) {
            std::error_code ec;
            fs::remove_all(runtime_dir, ec);
            if (ec && fs::exists(runtime_dir)) {
                throw std::runtime_error("Failed to remove " + runtime_dir + ": " + ec.message());
            }
        }
    }

}

// ============================================================================
// Query operations
// ============================================================================

std::string BackendManager::get_latest_version(const std::string& recipe, const std::string& backend) {
    try {
        return get_version_from_config(recipe, normalize_backend_name(recipe, backend));
    } catch (...) {
        return "";
    }
}

json BackendManager::get_all_backends_status() {
    auto statuses = SystemInfo::get_all_recipe_statuses();
    json result = json::array();

    for (const auto& recipe : statuses) {
        json recipe_json;
        recipe_json["recipe"] = recipe.name;

        json backends_json = json::array();
        for (const auto& backend : recipe.backends) {
            json b;
            b["name"] = backend.name;
            b["state"] = backend.state;
            b["message"] = backend.message;
            b["action"] = backend.action;
            if (!backend.version.empty()) {
                b["version"] = backend.version;
            }

            // Add release URL
            std::string release_url = get_release_url(recipe.name, backend.name);
            if (!release_url.empty()) {
                b["release_url"] = release_url;
            }

            backends_json.push_back(b);
        }
        recipe_json["backends"] = backends_json;
        result.push_back(recipe_json);
    }

    return result;
}

std::string BackendManager::get_release_url(const std::string& recipe, const std::string& backend) {
    try {
        std::string resolved_backend = normalize_backend_name(recipe, backend);

        if (recipe == "flm") {
            std::string version = get_latest_version(recipe, resolved_backend);
            if (!version.empty()) {
                return "https://github.com/FastFlowLM/FastFlowLM/releases/tag/" + version;
            }
            return "";
        }

        auto params = get_install_params(recipe, resolved_backend);
        return "https://github.com/" + params.repo + "/releases/tag/" + params.version;
    } catch (...) {
        return "";
    }
}

std::string BackendManager::get_download_filename(const std::string& recipe, const std::string& backend) {
    try {
        auto params = get_install_params(recipe, normalize_backend_name(recipe, backend));
        return params.filename;
    } catch (...) {
        return "";
    }
}

BackendManager::BackendEnrichment BackendManager::get_backend_enrichment(const std::string& recipe, const std::string& backend) {
    BackendEnrichment result;
    try {
        std::string resolved_backend = normalize_backend_name(recipe, backend);

        if (recipe == "flm") {
            result.version = get_latest_version(recipe, resolved_backend);
            if (!result.version.empty()) {
                result.release_url = "https://github.com/FastFlowLM/FastFlowLM/releases/tag/" + result.version;
            }
            // FLM installer artifact used by install_flm_if_needed().
            result.download_filename = "flm-setup.exe";
            return result;
        }

        // All standard recipes (including ryzenai-llm): one get_install_params() call gives us everything
        auto params = get_install_params(recipe, resolved_backend);
        result.release_url = "https://github.com/" + params.repo + "/releases/tag/" + params.version;
        result.download_filename = params.filename;
        result.version = params.version;
    } catch (...) {}
    return result;
}

} // namespace lemon
