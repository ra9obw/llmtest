

response = "I need to make sure not to repeat the code, just the doc-string. Also, avoid unnecessary tags. Let me structure it step by step. \
Start with @brief: Initializes the device, setting up attributes and the ADC hardware. \
@details: Describe the process, like getting device properties, initializing attributes, creating windows, handling exceptions, etc. \
@throws: VME::VMEException if initialization fails. \
@note: Mention the fallback mode check and the not_first_init_flag logic. \
Other tags like @param and @return are not needed here since the method is void and has no parameters.   \
Also, check if there are any other exceptions, but the code only catches VME::VMEException, so that's the only one to mention.\
Now, putting it all together in the doc-string format.\
</think>\
/**\
 * @brief Initializes the device, setting up attributes and the ADC hardware.\
 *\
 * @details This method is called during device initialization. It retrieves device properties from the database,\
 * initializes various attributes (such as version, device ID, status registers, etc.), and prepares the ADC hardware\
 * for operation. If the device is in fallback mode, a single default window is created. Otherwise, multiple windows\
 * are initialized based on the `numberOfWindows` property. The method also attempts to connect to the ADC hardware\
 * via the VME bus, initializing it with default settings. In case of failure, the device state is set to FAULT with\
 * an appropriate error message.\
 *\
 * @throws VME::VMEException if the ADC hardware initialization fails.\
 *\
 * @note The method checks if the device is being initialized for the first time by examining the size of the\
 * `windows_vector`. If it's empty, the method proceeds to create the initial window(s). Otherwise, it skips this\
 * step, assuming the device has already been initialized.\
 *\
 * @note The fallback mode is determined by comparing `numberOfWindows` with `default_window_nuber`. If they are\
 * equal, the device is considered to be in fallback mode, and a single default window is created.       \
 */\
"

# think_tag = "</think>"
# if think_tag in response:
#     parts = response.split(think_tag, 1)
#     if len(parts) > 1:
#         print(parts[1].strip())

# import re
# def extract_docstrings(text):
#     """
#     Извлекает все докстринги из переданного текста.
    
#     Аргументы:
#         text (str): Исходный текст, который может содержать докстринги.
        
#     Возвращает:
#         str: Текст, содержащий только докстринги, или пустую строку, если докстрингов нет.
#     """
#     # Регулярное выражение для поиска многострочных докстрингов (Python и C/C++ стиль)
#     pattern = r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|/\*\*[\s\S]*?\*/)'
#     docstrings = re.findall(pattern, text)
#     return '\n'.join(docstrings) if docstrings else ''

# print(extract_docstrings(response))



# import os
# import json
# from pathlib import Path
from clang.cindex import Index, CursorKind, TranslationUnit, Config

crsrs = [
CursorKind.CLASS_DECL,
CursorKind.STRUCT_DECL,
CursorKind.CLASS_TEMPLATE,
CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
CursorKind.NAMESPACE,
CursorKind.CXX_METHOD,
CursorKind.CONSTRUCTOR,
CursorKind.DESTRUCTOR,
CursorKind.FUNCTION_DECL,
CursorKind.FUNCTION_TEMPLATE,
CursorKind.LAMBDA_EXPR,
CursorKind.PREPROCESSING_DIRECTIVE,
CursorKind.MACRO_DEFINITION,
CursorKind.CONVERSION_FUNCTION,
CursorKind.TYPE_ALIAS_DECL,
CursorKind.TYPEDEF_DECL,
CursorKind.VAR_DECL,
CursorKind.FIELD_DECL,
CursorKind.ENUM_DECL,
CursorKind.ENUM_CONSTANT_DECL,
CursorKind.UNION_DECL,
CursorKind.USING_DIRECTIVE
]

for crsr in crsrs:
    print(crsr.name.lower())


# class_decl
# struct_decl
# class_template
# class_template_partial_specialization
# namespace
# cxx_method
# constructor
# destructor
# function_decl
# function_template
# lambda_expr
# preprocessing_directive
# macro_definition
# conversion_function
# type_alias_decl
# typedef_decl
# var_decl
# field_decl
# enum_decl
# enum_constant_decl
# union_decl
# using_directive

# # Укажи полный путь к libclang.dll
# LIBCLANG_DLL_PATH = r"E:\\Programs\\clang-llvm-windows-msvc\\bin\\libclang.dll"
# # Ставим путь к libclang.dll
# Config.set_library_file(LIBCLANG_DLL_PATH)
# # Теперь можно использовать clang
# index = Index.create()
# print("Clang успешно загружен")


            # print("Available CursorKind attributes:")
            # for name in dir(CursorKind):
            #     if not name.startswith('_') and name.isupper():
            #         print(name)


# print("TokenKinds: ")
#             for name in dir(TokenKind):
#                 if not name.startswith('_') and name.isupper():
#                     print(name)

# TokenKinds:
# COMMENT
# IDENTIFIER
# KEYWORD
# LITERAL
# PUNCTUATION

# CursorKinds:
# ADDR_LABEL_EXPR
# ALIGNED_ATTR
# ANNOTATE_ATTR
# ARRAY_SUBSCRIPT_EXPR
# ASM_LABEL_ATTR
# ASM_STMT
# BINARY_OPERATOR
# BLOCK_EXPR
# BREAK_STMT
# BUILTIN_BIT_CAST_EXPR
# CALL_EXPR
# CASE_STMT
# CHARACTER_LITERAL
# CLASS_DECL
# CLASS_TEMPLATE
# CLASS_TEMPLATE_PARTIAL_SPECIALIZATION
# COMPOUND_ASSIGNMENT_OPERATOR
# COMPOUND_LITERAL_EXPR
# COMPOUND_STMT
# CONCEPT_DECL
# CONCEPT_SPECIALIZATION_EXPR
# CONDITIONAL_OPERATOR
# CONSTRUCTOR
# CONST_ATTR
# CONTINUE_STMT
# CONVERGENT_ATTR
# CONVERSION_FUNCTION
# CSTYLE_CAST_EXPR
# CUDACONSTANT_ATTR
# CUDADEVICE_ATTR
# CUDAGLOBAL_ATTR
# CUDAHOST_ATTR
# CUDASHARED_ATTR
# CXX_ACCESS_SPEC_DECL
# CXX_ADDRSPACE_CAST_EXPR
# CXX_BASE_SPECIFIER
# CXX_BOOL_LITERAL_EXPR
# CXX_CATCH_STMT
# CXX_CONST_CAST_EXPR
# CXX_DELETE_EXPR
# CXX_DYNAMIC_CAST_EXPR
# CXX_FINAL_ATTR
# CXX_FOR_RANGE_STMT
# CXX_FUNCTIONAL_CAST_EXPR
# CXX_METHOD
# CXX_NEW_EXPR
# CXX_NULL_PTR_LITERAL_EXPR
# CXX_OVERRIDE_ATTR
# CXX_PAREN_LIST_INIT_EXPR
# CXX_REINTERPRET_CAST_EXPR
# CXX_STATIC_CAST_EXPR
# CXX_THIS_EXPR
# CXX_THROW_EXPR
# CXX_TRY_STMT
# CXX_TYPEID_EXPR
# CXX_UNARY_EXPR
# DECL_REF_EXPR
# DECL_STMT
# DEFAULT_STMT
# DESTRUCTOR
# DLLEXPORT_ATTR
# DLLIMPORT_ATTR
# DO_STMT
# ENUM_CONSTANT_DECL
# ENUM_DECL
# FIELD_DECL
# FIXED_POINT_LITERAL
# FLAG_ENUM
# FLOATING_LITERAL
# FOR_STMT
# FRIEND_DECL
# FUNCTION_DECL
# FUNCTION_TEMPLATE
# GENERIC_SELECTION_EXPR
# GNU_NULL_EXPR
# GOTO_STMT
# IB_ACTION_ATTR
# IB_OUTLET_ATTR
# IB_OUTLET_COLLECTION_ATTR
# IF_STMT
# IMAGINARY_LITERAL
# INCLUSION_DIRECTIVE
# INDIRECT_GOTO_STMT
# INIT_LIST_EXPR
# INTEGER_LITERAL
# INVALID_CODE
# INVALID_FILE
# LABEL_REF
# LABEL_STMT
# LAMBDA_EXPR
# LINKAGE_SPEC
# MACRO_DEFINITION
# MACRO_INSTANTIATION
# MEMBER_REF
# MEMBER_REF_EXPR
# MODULE_IMPORT_DECL
# MS_ASM_STMT
# NAMESPACE
# NAMESPACE_ALIAS
# NAMESPACE_REF
# NODUPLICATE_ATTR
# NOT_IMPLEMENTED
# NO_DECL_FOUND
# NS_CONSUMED
# NS_CONSUMES_SELF
# NS_RETURNS_AUTORELEASED
# NS_RETURNS_NOT_RETAINED
# NS_RETURNS_RETAINED
# NULL_STMT
# OBJC_AT_CATCH_STMT
# OBJC_AT_FINALLY_STMT
# OBJC_AT_SYNCHRONIZED_STMT
# OBJC_AT_THROW_STMT
# OBJC_AT_TRY_STMT
# OBJC_AUTORELEASE_POOL_STMT
# OBJC_AVAILABILITY_CHECK_EXPR
# OBJC_BOXABLE
# OBJC_BRIDGE_CAST_EXPR
# OBJC_CATEGORY_DECL
# OBJC_CATEGORY_IMPL_DECL
# OBJC_CLASS_METHOD_DECL
# OBJC_CLASS_REF
# OBJC_DESIGNATED_INITIALIZER
# OBJC_DYNAMIC_DECL
# OBJC_ENCODE_EXPR
# OBJC_EXCEPTION
# OBJC_EXPLICIT_PROTOCOL_IMPL
# OBJC_FOR_COLLECTION_STMT
# OBJC_IMPLEMENTATION_DECL
# OBJC_INDEPENDENT_CLASS
# OBJC_INSTANCE_METHOD_DECL
# OBJC_INTERFACE_DECL
# OBJC_IVAR_DECL
# OBJC_MESSAGE_EXPR
# OBJC_NSOBJECT
# OBJC_PRECISE_LIFETIME
# OBJC_PROPERTY_DECL
# OBJC_PROTOCOL_DECL
# OBJC_PROTOCOL_EXPR
# OBJC_PROTOCOL_REF
# OBJC_REQUIRES_SUPER
# OBJC_RETURNS_INNER_POINTER
# OBJC_ROOT_CLASS
# OBJC_RUNTIME_VISIBLE
# OBJC_SELECTOR_EXPR
# OBJC_STRING_LITERAL
# OBJC_SUBCLASSING_RESTRICTED
# OBJC_SUPER_CLASS_REF
# OBJC_SYNTHESIZE_DECL
# OBJ_BOOL_LITERAL_EXPR
# OBJ_SELF_EXPR
# OMP_ARRAY_SECTION_EXPR
# OMP_ARRAY_SHAPING_EXPR
# OMP_ATOMIC_DIRECTIVE
# OMP_BARRIER_DIRECTIVE
# OMP_CANCELLATION_POINT_DIRECTIVE
# OMP_CANCEL_DIRECTIVE
# OMP_CANONICAL_LOOP
# OMP_CRITICAL_DIRECTIVE
# OMP_DEPOBJ_DIRECTIVE
# OMP_DISPATCH_DIRECTIVE
# OMP_DISTRIBUTE_DIRECTIVE
# OMP_DISTRIBUTE_PARALLELFOR_DIRECTIVE
# OMP_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE
# OMP_DISTRIBUTE_SIMD_DIRECTIVE
# OMP_ERROR_DIRECTIVE
# OMP_FLUSH_DIRECTIVE
# OMP_FOR_DIRECTIVE
# OMP_FOR_SIMD_DIRECTIVE
# OMP_GENERIC_LOOP_DIRECTIVE
# OMP_INTEROP_DIRECTIVE
# OMP_ITERATOR_EXPR
# OMP_MASKED_DIRECTIVE
# OMP_MASKED_TASK_LOOP_DIRECTIVE
# OMP_MASKED_TASK_LOOP_SIMD_DIRECTIVE
# OMP_MASTER_DIRECTIVE
# OMP_MASTER_TASK_LOOP_DIRECTIVE
# OMP_MASTER_TASK_LOOP_SIMD_DIRECTIVE
# OMP_META_DIRECTIVE
# OMP_ORDERED_DIRECTIVE
# OMP_PARALLELFORSIMD_DIRECTIVE
# OMP_PARALLEL_DIRECTIVE
# OMP_PARALLEL_FOR_DIRECTIVE
# OMP_PARALLEL_GENERIC_LOOP_DIRECTIVE
# OMP_PARALLEL_MASKED_DIRECTIVE
# OMP_PARALLEL_MASKED_TASK_LOOP_DIRECTIVE
# OMP_PARALLEL_MASKED_TASK_LOOP_SIMD_DIRECTIVE
# OMP_PARALLEL_MASTER_DIRECTIVE
# OMP_PARALLEL_MASTER_TASK_LOOP_DIRECTIVE
# OMP_PARALLEL_MASTER_TASK_LOOP_SIMD_DIRECTIVE
# OMP_PARALLEL_SECTIONS_DIRECTIVE
# OMP_SCAN_DIRECTIVE
# OMP_SCOPE_DIRECTIVE
# OMP_SECTIONS_DIRECTIVE
# OMP_SECTION_DIRECTIVE
# OMP_SIMD_DIRECTIVE
# OMP_SINGLE_DIRECTIVE
# OMP_TARGET_DATA_DIRECTIVE
# OMP_TARGET_DIRECTIVE
# OMP_TARGET_ENTER_DATA_DIRECTIVE
# OMP_TARGET_EXIT_DATA_DIRECTIVE
# OMP_TARGET_PARALLELFOR_DIRECTIVE
# OMP_TARGET_PARALLEL_DIRECTIVE
# OMP_TARGET_PARALLEL_FOR_SIMD_DIRECTIVE
# OMP_TARGET_PARALLEL_GENERIC_LOOP_DIRECTIVE
# OMP_TARGET_SIMD_DIRECTIVE
# OMP_TARGET_TEAMS_DIRECTIVE
# OMP_TARGET_TEAMS_DISTRIBUTE_DIRECTIVE
# OMP_TARGET_TEAMS_DISTRIBUTE_PARALLEL_FOR_DIRECTIVE
# OMP_TARGET_TEAMS_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE
# OMP_TARGET_TEAMS_DISTRIBUTE_SIMD_DIRECTIVE
# OMP_TARGET_TEAMS_GENERIC_LOOP_DIRECTIVE
# OMP_TARGET_UPDATE_DIRECTIVE
# OMP_TASKGROUP_DIRECTIVE
# OMP_TASKWAIT_DIRECTIVE
# OMP_TASKYIELD_DIRECTIVE
# OMP_TASK_DIRECTIVE
# OMP_TASK_LOOP_DIRECTIVE
# OMP_TASK_LOOP_SIMD_DIRECTIVE
# OMP_TEAMS_DIRECTIVE
# OMP_TEAMS_DISTRIBUTE_DIRECTIVE
# OMP_TEAMS_DISTRIBUTE_PARALLEL_FOR_DIRECTIVE
# OMP_TEAMS_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE
# OMP_TEAMS_DISTRIBUTE_SIMD_DIRECTIVE
# OMP_TEAMS_GENERIC_LOOP_DIRECTIVE
# OMP_TILE_DIRECTIVE
# OMP_UNROLL_DIRECTIVE
# OPEN_ACC_COMPUTE_DIRECTIVE
# OVERLOADED_DECL_REF
# OVERLOAD_CANDIDATE
# PACKED_ATTR
# PACK_EXPANSION_EXPR
# PACK_INDEXING_EXPR
# PAREN_EXPR
# PARM_DECL
# PREPROCESSING_DIRECTIVE
# PURE_ATTR
# REQUIRES_EXPR
# RETURN_STMT
# SEH_EXCEPT_STMT
# SEH_FINALLY_STMT
# SEH_LEAVE_STMT
# SEH_TRY_STMT
# SIZE_OF_PACK_EXPR
# STATIC_ASSERT
# STRING_LITERAL
# STRUCT_DECL
# SWITCH_STMT
# TEMPLATE_NON_TYPE_PARAMETER
# TEMPLATE_REF
# TEMPLATE_TEMPLATE_PARAMETER
# TEMPLATE_TYPE_PARAMETER
# TRANSLATION_UNIT
# TYPEDEF_DECL
# TYPE_ALIAS_DECL
# TYPE_ALIAS_TEMPLATE_DECL
# TYPE_REF
# UNARY_OPERATOR
# UNEXPOSED_ATTR
# UNEXPOSED_DECL
# UNEXPOSED_EXPR
# UNEXPOSED_STMT
# UNION_DECL
# USING_DECLARATION
# USING_DIRECTIVE
# VARIABLE_REF
# VAR_DECL
# VISIBILITY_ATTR
# WARN_UNUSED_ATTR
# WARN_UNUSED_RESULT_ATTR
# WHILE_STMT