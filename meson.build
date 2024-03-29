project('OpenGadget4','c','cpp', version: '4.0.0',
    default_options:
    ['optimization=3','buildtype=release','warning_level=3','cpp_std=c++17'])

# if get_option('buildtype')=='release'
#     add_global_arguments('-Ofast',language: 'cpp')
#     add_global_arguments('-DNDEBUG',language: 'cpp')
# else
#     add_global_arguments('-g3',language: 'cpp')
# endif

cpp=meson.get_compiler('cpp')
format = find_program('clang-format',required: false)
python = find_program('python3')
bash = find_program('bash')

mpi = dependency('mpi',language: 'cpp')
fftw3 = dependency('fftw3')
hdf5 = dependency('hdf5',language: 'cpp')
gsl = dependency('gsl')
hwloc = dependency('hwloc', required: false)
healpix = dependency('chealpix')
boost = dependency('boost', modules: ['mpi','serialization','unit_test_framework'])

latfield = dependency('latfield', fallback: ['LATfield2','liblatfield_dep'])
gevolution = dependency('gevolution', fallback: ['gevolution-1.2','libgevolution_dep'])

pkg=import('pkgconfig')

user_config_fname = get_option('config_file')
default_user_config_fname=meson.build_root() / 'Config.sh'


if user_config_fname==''
    user_config = files(default_user_config_fname)
else
    user_config = files(user_config_fname)
endif    

subdir('buildsystem')

sources=[]
headers=[]
subdir('src')

subdir('md5')
subdir('vectorclass')
subdir('half')

subdir('gadget')
gadget_exec = executable('Gadget4',
    sources,headers,config_header,compile_header,git_version,
    dependencies:
    [fftw3,mpi,hwloc,hdf5,gsl,md5,vectorclass,gadget,half,healpix,boost,
    latfield,gevolution],
    install: true)

if (format.found())
    run_target('format', 
    command: [format,'-i',sources,headers,
        md5_src,md5_headers,
        vectorclass_src,vectorclass_headers,
        gadget_src,gadget_headers,
        half_headers])
endif
