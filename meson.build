project('OpenGadget4','c','cpp', version: '4.0.0',
    default_options:
    ['optimization=3','buildtype=release','warning_level=3','cpp_std=c++17'])

if get_option('buildtype')=='release'
    add_global_arguments('-Ofast',language: 'cpp')
    add_global_arguments('-DNDEBUG',language: 'cpp')
endif

cpp=meson.get_compiler('cpp')
format = find_program('clang-format',required: false)
python = find_program('python3')
bash = find_program('bash')

mpi = dependency('mpi',language: 'cpp')
fftw3 = dependency('fftw3')
hdf5 = dependency('hdf5',language: 'cpp')
gsl = dependency('gsl')
hwloc = dependency('hwloc')

pkg=import('pkgconfig')

user_config=files('Config.sh')

subdir('buildsystem')

sources=[]
headers=[]
subdir('src')

subdir('md5')
subdir('vectorclass')


config_header = custom_target('config',
    build_by_default: true,
    input: user_config,
    output: ['gadgetconfig.h','compile_time_info.cc','compile_time_info_hdf5.cc'],
    command : [python,config_generator,'@INPUT@','@OUTDIR@'])

compile_header = custom_target('compileinfo',
    build_by_default: true,
    output: 'compiler-command-line-args.h',
    command : [python,compileflags_generator,'@OUTDIR@'])

git_version = custom_target('git-info',
    build_by_default: true,
    output: 'version.cc',
    command : [bash,gitinfo_generator,meson.source_root(),'@OUTDIR@'])


subdir('gadget')
executable('Gadget4',
    sources,headers,config_header,compile_header,git_version,
    dependencies: [fftw3,mpi,hwloc,hdf5,gsl,md5,vectorclass,gadget],
    install: true)

if (format.found())
    run_target('format', 
    command: [format,'-i',sources,headers,
        md5_src,md5_headers,
        vectorclass_src,vectorclass_headers,
        gadget_src,gadget_headers])
endif
