config_generator = files('config.py')
compileflags_generator = files('config-cflags.py')
gitinfo_generator = files('git_version.sh')
version_template = files('version.template')

git_version = custom_target('git-info',
    build_by_default: true,
    input : version_template,
    output: 'version.h',
    command : [bash,gitinfo_generator,meson.source_root(),'@OUTDIR@','@INPUT@'])

config_header = custom_target('config',
    build_by_default: true,
    input: user_config,
    output: ['gadgetconfig.h','compile_time_info.cc','compile_time_info_hdf5.cc'],
    command : [python,config_generator,'@INPUT@','@OUTDIR@'])

compile_header = custom_target('compileinfo',
    build_by_default: true,
    output: 'compiler-command-line-args.h',
    command : [python,compileflags_generator,'@OUTDIR@'])
