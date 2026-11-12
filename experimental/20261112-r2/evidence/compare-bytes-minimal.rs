pub unsafe fn compare(p: *const std::ffi::c_char) -> bool {
    std::ffi::CStr::from_ptr(p).to_bytes() == b"x".as_slice()
}
