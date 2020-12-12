subdir('include')
subdir('src')
vectorclass_lib=static_library('vectorclass',vectorclass_src+vectorclass_headers,
    include_directories: vectorclass_inc,
    install : true)

install_headers(vectorclass_headers,subdir : 'vectorclass')
vectorclass = declare_dependency(include_directories: vectorclass_inc,
    link_with : vectorclass_lib)

pkg.generate(vectorclass_lib,
    description: 'Vectorized Operations C++')
