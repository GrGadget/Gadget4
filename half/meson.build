subdir('include')

half_lib=library('half',half_headers,
    include_directories: half_inc,
    install : true)

install_headers(half_headers,subdir: 'half')
half = declare_dependency(include_directories: half_inc,
    link_with : half_lib)

pkg.generate(half_lib,
    description: 'Half precision library')
