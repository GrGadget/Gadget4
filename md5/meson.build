subdir('include')
subdir('src')
md5_lib=static_library('md5',md5_src+md5_headers,
    include_directories: md5_inc,
    install : true)

install_headers(md5_headers,subdir: 'md5')
md5 = declare_dependency(include_directories: md5_inc,
    link_with : md5_lib)

pkg.generate(md5_lib,
    description: 'MD5 hash utility')
