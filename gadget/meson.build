subdir('include')
subdir('src')
gadget_lib=static_library('gadget',gadget_src+gadget_headers,
    config_header[0],
    include_directories: gadget_inc,
    install : true)

install_headers(gadget_headers,subdir: 'gagdet')
gadget = declare_dependency(include_directories: gadget_inc,
    link_with : gadget_lib)

pkg.generate(gadget_lib,
    description: 'Gadget tools')
