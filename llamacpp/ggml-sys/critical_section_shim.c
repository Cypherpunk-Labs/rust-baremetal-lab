// Host-only no-op critical-section hooks for ggml (ggml-threading.cpp is not
// compiled; the kernel is single-threaded). On the bare-metal target these are
// provided by `os-api` instead, so this file is only compiled for the host.
void ggml_critical_section_start(void) {}
void ggml_critical_section_end(void) {}
