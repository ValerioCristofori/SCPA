set_source_files_properties(utils.c PROPERTIES LANGUAGE CXX )
set_source_files_properties(preprocessing.c PROPERTIES LANGUAGE CXX )
set_source_files_properties(mmio.c PROPERTIES LANGUAGE CXX )
set(lib_files
        preprocessing.c
        utils.c
        utils.h
        mmio.c
        mmio.h
        )
add_library(lib ${lib_files})