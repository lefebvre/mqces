# mqces command-line interface.

add_executable(mqces_cli mqces_cli.cpp)
target_link_libraries(mqces_cli
    PRIVATE
        mqces::mqces
        mqces::warnings
        nlohmann_json::nlohmann_json
)
set_target_properties(mqces_cli PROPERTIES OUTPUT_NAME mqces)
mqces_target_clang_tidy(mqces_cli)

if(MQCES_ENABLE_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS mqces_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
