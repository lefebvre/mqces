# mqces command-line interface.

add_executable(mqces_cli mqces_cli.cpp)
target_link_libraries(mqces_cli
    PRIVATE
        mqces::mqces
        mqces::warnings
        nlohmann_json::nlohmann_json
)
set_target_properties(mqces_cli PROPERTIES OUTPUT_NAME mqces)
