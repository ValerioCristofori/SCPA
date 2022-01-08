set_source_files_properties(utils.c PROPERTIES LANGUAGE CXX )
set_source_files_properties(mmio.c PROPERTIES LANGUAGE CXX )
set(lib_files
        utils.c
        utils.h
        mmio.c
        mmio.h
        )
add_library(libraries ${lib_files})