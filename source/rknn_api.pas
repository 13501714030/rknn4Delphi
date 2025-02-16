{**************************************************************************

rknn4Delphi (Rockchip RKNN Inference Frame Delphi Wrapped

author: Tom Yea
create time: 2025/1/25
contact: tom_ye@qq.com

this .pas is rknn API reference

****************************************************************************}


unit rknn_api;

interface

uses
  System.SysUtils, System.Types, System.IOUtils, System.Math;

const
  // 设置高优先级上下文
  RKNN_FLAG_PRIOR_HIGH = $00000000;
  // 设置中优先级上下文
  RKNN_FLAG_PRIOR_MEDIUM = $00000001;
  // 设置低优先级上下文
  RKNN_FLAG_PRIOR_LOW = $00000002;
  // 异步模式。启用时，rknn_outputs_get 不会阻塞太久，因为它直接获取前一帧的结果，可以提高单线程模式下的帧率，
  // 但代价是 rknn_outputs_get 不会获取当前帧的结果。在多线程模式下不需要开启此模式。
  RKNN_FLAG_ASYNC_MASK = $00000004;
  // 收集性能模式。启用时，可以通过 rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, ...) 获取详细的性能报告，但会降低帧率。
  RKNN_FLAG_COLLECT_PERF_MASK = $00000008;
  // 在外部分配所有内存，包括权重/内部/输入/输出
  RKNN_FLAG_MEM_ALLOC_OUTSIDE = $00000010;
  // 共享相同网络结构的权重内存
  RKNN_FLAG_SHARE_WEIGHT_MEM = $00000020;
  // 从外部发送 fence fd
  RKNN_FLAG_FENCE_IN_OUTSIDE = $00000040;
  // 从内部获取 fence fd
  RKNN_FLAG_FENCE_OUT_OUTSIDE = $00000080;
  // 仅收集模型信息：只能通过 rknn_query 获取 total_weight_size 和 total_internal_size
  RKNN_FLAG_COLLECT_MODEL_INFO_ONLY = $00000100;
  // 在外部分配内部内存
  RKNN_FLAG_INTERNAL_ALLOC_OUTSIDE = $00000200;
  // 当 NPU 不支持操作符时，设置 GPU 为优先执行后端
  RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU = $00000400;
  // 启用分配 SRAM 类型缓冲区
  RKNN_FLAG_ENABLE_SRAM = $00000800;
  // SRAM 类型缓冲区在不同上下文之间共享
  RKNN_FLAG_SHARE_SRAM = $00001000;
  // 默认优先级为 -19，此标志可以禁用默认优先级
  RKNN_FLAG_DISABLE_PROC_HIGH_PRIORITY = $00002000;
  // 不刷新输入缓冲区缓存，用户必须确保在调用 rknn_run 之前输入张量已刷新缓存。!!! 在调用 rknn_inputs_set() 设置输入数据时不要使用此标志。
  RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE = $00004000;
  // 不使输出缓冲区缓存无效。用户不能直接访问 output_mem->virt_addr，否则会导致缓存一致性问题。如果要使用 output_mem->virt_addr，必须使用 rknn_mem_sync (ctx, mem, RKNN_MEMORY_SYNC_FROM_DEVICE) 刷新缓存。此标志通常用于 NPU 的输出数据不被 CPU 访问，而是被 GPU 或 RGA 访问以减少刷新缓存所需的时间。!!! 在调用 rknn_outputs_get() 获取输出数据时不要使用此标志。
  RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE = $00008000;
  // 当模型数据缓冲区由 NPU 分配时使用此标志，NPU 可以直接访问该缓冲区。
  RKNN_FLAG_MODEL_BUFFER_ZERO_COPY = $00010000;
  // 此标志是内存分配标志，在 rknn_create_mem2() 中在没有上下文时使用。
  RKNN_MEM_FLAG_ALLOC_NO_CONTEXT = $00020000;

  // RKNN API 返回的错误码
  RKNN_SUCC = 0; // 执行成功
  RKNN_ERR_FAIL = -1; // 执行失败
  RKNN_ERR_TIMEOUT = -2; // 执行超时
  RKNN_ERR_DEVICE_UNAVAILABLE = -3; // 设备不可用
  RKNN_ERR_MALLOC_FAIL = -4; // 内存分配失败
  RKNN_ERR_PARAM_INVALID = -5; // 参数无效
  RKNN_ERR_MODEL_INVALID = -6; // 模型无效
  RKNN_ERR_CTX_INVALID = -7; // 上下文无效
  RKNN_ERR_INPUT_INVALID = -8; // 输入无效
  RKNN_ERR_OUTPUT_INVALID = -9; // 输出无效
  RKNN_ERR_DEVICE_UNMATCH = -10; // 设备不匹配，请更新 rknn SDK 和 NPU 驱动/固件
  RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL = -11; // 此 RKNN 模型使用预编译模式，但与当前驱动不兼容
  RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION = -12; // 此 RKNN 模型设置了优化级别，但与当前驱动不兼容
  RKNN_ERR_TARGET_PLATFORM_UNMATCH = -13; // 此 RKNN 模型设置了目标平台，但与当前平台不兼容

  // 张量的定义
  RKNN_MAX_DIMS = 16; // 张量的最大维度
  RKNN_MAX_NUM_CHANNEL = 15; // 输入张量的最大通道数
  RKNN_MAX_NAME_LEN = 256; // 张量的最大名称长度
  RKNN_MAX_DYNAMIC_SHAPE_NUM = 512; // 每个输入的最大动态形状数量

type
  // 定义 rknn_context 类型
  {$IFDEF ARM}
    rknn_context = UInt32;
  {$ELSE}
    rknn_context = UInt64;
  {$ENDIF}
  Prknn_context = ^rknn_context;

  // rknn_query 的查询命令
  rknn_query_cmd = (
    RKNN_QUERY_IN_OUT_NUM = 0, // 查询输入和输出张量的数量
    RKNN_QUERY_INPUT_ATTR = 1, // 查询输入张量的属性
    RKNN_QUERY_OUTPUT_ATTR = 2, // 查询输出张量的属性
    RKNN_QUERY_PERF_DETAIL = 3, // 查询详细性能，需要在调用 rknn_init 时设置 RKNN_FLAG_COLLECT_PERF_MASK，此查询在 rknn_outputs_get 后有效
    RKNN_QUERY_PERF_RUN = 4, // 查询运行时间，此查询在 rknn_outputs_get 后有效
    RKNN_QUERY_SDK_VERSION = 5, // 查询 SDK 和驱动版本
    RKNN_QUERY_MEM_SIZE = 6, // 查询权重和内部内存大小
    RKNN_QUERY_CUSTOM_STRING = 7, // 查询自定义字符串
    RKNN_QUERY_NATIVE_INPUT_ATTR = 8, // 查询原生输入张量的属性
    RKNN_QUERY_NATIVE_OUTPUT_ATTR = 9, // 查询原生输出张量的属性
    RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR = 8, // 查询原生输入张量的属性
    RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR = 9, // 查询原生输出张量的属性
    RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR = 10, // 查询原生输入张量的属性
    RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR = 11, // 查询原生输出张量的属性
    RKNN_QUERY_DEVICE_MEM_INFO = 12, // 查询 rknn 内存信息的属性
    RKNN_QUERY_INPUT_DYNAMIC_RANGE = 13, // 查询 rknn 输入张量的动态形状范围
    RKNN_QUERY_CURRENT_INPUT_ATTR = 14, // 查询 rknn 输入张量的当前形状，仅对动态 rknn 模型有效
    RKNN_QUERY_CURRENT_OUTPUT_ATTR = 15, // 查询 rknn 输出张量的当前形状，仅对动态 rknn 模型有效
    RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR = 16, // 查询 rknn 输入张量的当前原生形状，仅对动态 rknn 模型有效
    RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR = 17, // 查询 rknn 输出张量的当前原生形状，仅对动态 rknn 模型有效
    RKNN_QUERY_CMD_MAX
  );

  rknn_sdk_version = record
    api_version: array [0 .. 255] of Char;
    drv_version: array [0 .. 255] of Char;
  end;

  // 张量数据类型
  rknn_tensor_type = (
    RKNN_TENSOR_FLOAT32 = 0, // 数据类型为 float32
    RKNN_TENSOR_FLOAT16, // 数据类型为 float16
    RKNN_TENSOR_INT8, // 数据类型为 int8
    RKNN_TENSOR_UINT8, // 数据类型为 uint8
    RKNN_TENSOR_INT16, // 数据类型为 int16
    RKNN_TENSOR_UINT16, // 数据类型为 uint16
    RKNN_TENSOR_INT32, // 数据类型为 int32
    RKNN_TENSOR_UINT32, // 数据类型为 uint32
    RKNN_TENSOR_INT64, // 数据类型为 int64
    RKNN_TENSOR_BOOL, // 数据类型为布尔值
    RKNN_TENSOR_INT4, // 数据类型为 int4
    RKNN_TENSOR_BFLOAT16, // 数据类型为 bfloat16
    RKNN_TENSOR_TYPE_MAX
  );

  // 量化类型
  rknn_tensor_qnt_type = (
    RKNN_TENSOR_QNT_NONE = 0, // 无量化
    RKNN_TENSOR_QNT_DFP, // 动态定点量化
    RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC, // 非对称仿射量化
    RKNN_TENSOR_QNT_MAX
  );

  // 张量数据格式
  rknn_tensor_format = (
    RKNN_TENSOR_NCHW = 0, // 数据格式为 NCHW
    RKNN_TENSOR_NHWC, // 数据格式为 NHWC
    RKNN_TENSOR_NC1HWC2, // 数据格式为 NC1HWC2
    RKNN_TENSOR_UNDEFINED, // 未定义的数据格式
    RKNN_TENSOR_FORMAT_MAX
  );

  // 目标 NPU 核心的运行模式
  rknn_core_mask = (
    RKNN_NPU_CORE_AUTO = 0, // 默认，随机在 NPU 核心上运行
    RKNN_NPU_CORE_0 = 1, // 在 NPU 核心 0 上运行
    RKNN_NPU_CORE_1 = 2, // 在 NPU 核心 1 上运行
    RKNN_NPU_CORE_2 = 4, // 在 NPU 核心 2 上运行
    RKNN_NPU_CORE_0_1 = RKNN_NPU_CORE_0 or RKNN_NPU_CORE_1, // 在 NPU 核心 0 和 1 上运行
    RKNN_NPU_CORE_0_1_2 = RKNN_NPU_CORE_0_1 or RKNN_NPU_CORE_2, // 在 NPU 核心 0、1 和 2 上运行
    RKNN_NPU_CORE_ALL = $FFFF, // 自动选择，根据平台选择多个 NPU 核心运行
    RKNN_NPU_CORE_UNDEFINED
  );

  // RKNN_QUERY_IN_OUT_NUM 的信息
  rknn_input_output_num = record
    n_input: UInt32; // 输入数量
    n_output: UInt32; // 输出数量
  end;
  Prknn_input = ^rknn_input;

  // RKNN_QUERY_INPUT_ATTR / RKNN_QUERY_OUTPUT_ATTR 的信息
  rknn_tensor_attr = record
    index: UInt32;                                      // 4bytes 输入参数，输入/输出张量的索引，需要在调用 rknn_query 前设置
    n_dims: UInt32;                                     // 4bytes 维度数量
    dims: array [0..RKNN_MAX_DIMS - 1] of UInt32;       // 16x4=64bytes 维度数组
    name: array [0..RKNN_MAX_NAME_LEN - 1] of AnsiChar; // 256bytes 张量名称
    n_elems: UInt32;                                    // 4bytes 元素数量
    size: UInt32;                                       // 4bytes 张量的字节大小
    fmt: Integer; //rknn_tensor_format;                            // 1bytes 张量的数据格式
    type_: Integer; //rknn_tensor_type;                            // 1bytes 张量的数据类型
    qnt_type: Integer; //rknn_tensor_qnt_type;                     // 1bytes 张量的量化类型
    fl: Int8;                                           // 1bytes RKNN_TENSOR_QNT_DFP 的小数长度
    zp: Int32;                                          // 4bytes RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC 的零点
    scale: Float32;                                     // 4bytes RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC 的缩放比例
    w_stride: UInt32;                                   // 4bytes 输入张量沿宽度维度的步幅，只读，0 表示等于宽度
    size_with_stride: UInt32;                           // 4bytes 带步幅的张量字节大小
    pass_through: UInt8;                                // 1bytes 直通模式，用于 rknn_set_io_mem 接口。如果为 TRUE，buf 数据直接传递给 rknn 模型的输入节点，无需任何转换，以下变量无需设置。如果为 FALSE，buf 数据根据以下类型和格式转换为与模型一致的输入，因此需要设置以下变量。
    h_stride: UInt32;                                   // 4bytes 输入张量沿高度维度的步幅，只写，如果设置为 0，h_stride = height
  end;
  Prknn_tensor_attr = ^rknn_tensor_attr;

  // 输入动态范围信息
  rknn_input_range = record
    index: UInt32; // 输入参数，输入/输出张量的索引，需要在调用 rknn_query 前设置
    shape_number: UInt32; // 形状数量
    fmt: rknn_tensor_format; // 张量的数据格式
    name: array[0..RKNN_MAX_NAME_LEN - 1] of AnsiChar; // 张量名称
    dyn_range: array[0..RKNN_MAX_DYNAMIC_SHAPE_NUM - 1, 0..RKNN_MAX_DIMS - 1] of UInt32; // 动态输入维度范围
    n_dims: UInt32; // 维度数量
  end;

  // RKNN_QUERY_PERF_DETAIL 的信息
  rknn_perf_detail = record
    perf_data: PAnsiChar; // 性能详细信息的字符串指针，用户无需释放
    data_len: UInt64; // 字符串长度
  end;

  // RKNN_QUERY_PERF_RUN 的信息
  rknn_perf_run = record
    run_duration: Int64; // 实际推理时间（微秒）
  end;

  // RKNN_QUERY_MEM_SIZE 的信息
  rknn_mem_size = record
    total_weight_size: UInt32; // 权重内存大小
    total_internal_size: UInt32; // 内部内存大小，不包括输入/输出
    total_dma_allocated_size: UInt64; // 分配的 DMA 内存总大小
    total_sram_size: UInt32; // 为 rknn 保留的系统 SRAM 总大小
    free_sram_size: UInt32; // 为 rknn 保留的系统 SRAM 空闲大小
    reserved: array[0..9] of UInt32; // 保留字段
  end;

  // RKNN_QUERY_CUSTOM_STRING 的信息
  rknn_custom_string = record
    string_: array[0..1023] of AnsiChar; // 自定义字符串，最大长度为 1024 字节
  end;

  // rknn_tensor_mem 的标志
  rknn_tensor_mem_flags = (
    RKNN_TENSOR_MEMORY_FLAGS_ALLOC_INSIDE = 1, // 用于标记在 rknn_destroy_mem() 中是否需要释放 "mem" 指针本身。如果设置了 RKNN_TENSOR_MEMORY_FLAGS_ALLOC_INSIDE 标志，rknn_destroy_mem() 将调用 free(mem)。
    RKNN_TENSOR_MEMORY_FLAGS_FROM_FD = 2, // 用于标记在 rknn_create_mem_from_fd() 中是否需要释放 "mem" 指针本身。如果设置了 RKNN_TENSOR_MEMORY_FLAGS_FROM_FD 标志，rknn_destroy_mem() 将调用 free(mem)。
    RKNN_TENSOR_MEMORY_FLAGS_FROM_PHYS = 3, // 用于标记在 rknn_create_mem_from_phys() 中是否需要释放 "mem" 指针本身。如果设置了 RKNN_TENSOR_MEMORY_FLAGS_FROM_PHYS 标志，rknn_destroy_mem() 将调用 free(mem)。
    RKNN_TENSOR_MEMORY_FLAGS_UNKNOWN
  );

  // 同步可缓存 rknn 内存的模式
  rknn_mem_sync_mode = (
    RKNN_MEMORY_SYNC_TO_DEVICE = $1, // 用于在 CPU 访问数据后确保设备访问数据的一致性
    RKNN_MEMORY_SYNC_FROM_DEVICE = $2, // 用于在设备访问数据后确保 CPU 访问数据的一致性
    RKNN_MEMORY_SYNC_BIDIRECTIONAL = RKNN_MEMORY_SYNC_TO_DEVICE or RKNN_MEMORY_SYNC_FROM_DEVICE // 用于确保设备和 CPU 之间双向数据访问的一致性
  );

  // 张量的内存信息
  rknn_tensor_mem = record
    virt_addr: Pointer; // 张量缓冲区的虚拟地址
    phys_addr: UInt64; // 张量缓冲区的物理地址
    fd: Int32; // 张量缓冲区的文件描述符
    offset: Int32; // 内存的偏移量
    size: UInt32; // 张量缓冲区的大小
    flags: UInt32; // 张量缓冲区的标志，保留
    priv_data: Pointer; // 张量缓冲区的私有数据
  end;
  Prknn_tensor_mem = ^rknn_tensor_mem;

  // rknn_input_set 的输入信息
  rknn_input = record
    index: UInt32; // 输入索引
    buf: Pointer; // 输入的缓冲区
    size: UInt32; // 输入缓冲区的大小
    pass_through: UInt8; // 直通模式。如果为 TRUE，buf 数据直接传递给 rknn 模型的输入节点，无需任何转换，以下变量无需设置。如果为 FALSE，buf 数据根据以下类型和格式转换为与模型一致的输入，因此需要设置以下变量。
    type_: Integer; // 输入缓冲区的数据类型
    fmt: Integer; // 输入缓冲区的数据格式。目前 NPU 的内部输入格式默认为 NCHW，因此输入 NCHW 数据可以避免驱动中的格式转换。
  end;
//  Prknn_input = ^rknn_input;

  // rknn_outputs_get 的输出信息
  rknn_output = record
    want_float: UInt8; // 是否将输出数据转换为浮点数
    is_prealloc: UInt8; // 缓冲区是否预分配。如果为 TRUE，需要设置以下变量。如果为 FALSE，无需设置以下变量。
    index: UInt32; // 输出索引
    buf: Pointer; // 输出的缓冲区。当 is_prealloc = FALSE 且调用 rknn_outputs_release 时，此 buf 指针将被释放，之后不能再使用。
    size: UInt32; // 输出缓冲区的大小
  end;
  Prknn_output = ^rknn_output;

  rknn_outputs = array of rknn_output;

  // rknn_init 的扩展信息
  rknn_init_extend = record
    ctx: rknn_context; // rknn 上下文
    real_model_offset: Int32; // 真实的 rknn 模型文件偏移量，仅在通过 rknn 文件路径和零拷贝模型初始化上下文时有效
    real_model_size: UInt32; // 真实的 rknn 模型文件大小，仅在通过 rknn 文件路径和零拷贝模型初始化上下文时有效
    model_buffer_fd: Int32; // 模型缓冲区的文件描述符
    model_buffer_flags: UInt32; // 模型缓冲区的标志
    reserved: array[0..111] of UInt8; // 保留字段
  end;
  Prknn_init_extend = ^rknn_init_extend;

  // rknn_run 的扩展信息
  rknn_run_extend = record
    frame_id: UInt64; // 输出参数，指示当前运行的帧 ID
    non_block: Int32; // 运行的阻塞标志，0 为阻塞，1 为非阻塞
    timeout_ms: Int32; // 阻塞模式的超时时间，单位为毫秒
    fence_fd: Int32; // 来自其他单元的 fence fd
  end;
  Prknn_run_extend = ^rknn_run_extend;

  // rknn_outputs_get 的扩展信息
  rknn_output_extend = record
    frame_id: UInt64; // 输出参数，指示输出的帧 ID，对应于 rknn_run_extend.frame_id
  end;
  Prknn_output_extend = ^rknn_output_extend;

  TboxArray = array [0 .. 3] of Single;

  ///////////////////////////////////////////////////////////////////////////////////////////

  letterbox_t = record
    x_pad: Integer;
    y_pad: Integer;
    scale: Single;
  end;

  image_rect_t = record
    left: Integer;
    top: Integer;
    right: Integer;
    bottom: Integer;
  end;

  object_detect_result = record
    box: image_rect_t;
    prop: Single;
    cls_id: Integer;
  end;

  object_detect_result_list = record
    ID: Integer;
    Count: Integer;
    results: array [0 .. 127] of object_detect_result;
  end;

  rknn_app_context_t = record
    rknn_ctx: rknn_context;
    io_num: rknn_input_output_num;
    input_attrs: Prknn_tensor_attr;
    output_attrs: Prknn_tensor_attr;

    model_channel: Integer;
    model_width: Integer;
    model_height: Integer;
    is_quant: Boolean;
  end;

  //////////////////////////////////////////////////////////////////////////////////////////

  // rknn_init 函数声明
  Tknn_init = function(context: Prknn_context; model: Pointer; size: UInt32; flag: UInt32; extend: Prknn_init_extend): Int32; cdecl;

  // rknn_dup_context 函数声明
  Trknn_dup_context = function(context_in: Prknn_context; context_out: Prknn_context): Int32; cdecl;

  // rknn_destroy 函数声明
  Trknn_destroy = function(context: rknn_context): Int32; cdecl;

  // rknn_query 函数声明
  Trknn_query = function(context: rknn_context; cmd: rknn_query_cmd; info: Pointer; size: UInt32): Int32; cdecl;

  // rknn_inputs_set 函数声明
  Trknn_inputs_set = function(context: rknn_context; n_inputs: UInt32; inputs: Prknn_input): Int32; cdecl;

  // rknn_set_batch_core_num 函数声明
  Trknn_set_batch_core_num = function(context: rknn_context; core_num: Int32): Int32; cdecl;

  // rknn_set_core_mask 函数声明
  Trknn_set_core_mask = function(context: rknn_context; core_mask: rknn_core_mask): Int32; cdecl;

  // rknn_run 函数声明
  Trknn_run = function(context: rknn_context; extend: Prknn_run_extend): Int32; cdecl;

  // rknn_wait 函数声明
  Trknn_wait = function(context: rknn_context; extend: Prknn_run_extend): Int32; cdecl;

  // rknn_outputs_get 函数声明
  Trknn_outputs_get = function(context: rknn_context; n_outputs: UInt32; outputs: Prknn_output; extend: Prknn_output_extend): Int32; cdecl;

  // rknn_outputs_release 函数声明
  Trknn_outputs_release = function(context: rknn_context; n_outputs: UInt32; outputs: Prknn_output): Int32; cdecl;

  // rknn_create_mem_from_phys 函数声明
  Trknn_create_mem_from_phys = function(ctx: rknn_context; phys_addr: UInt64; virt_addr: Pointer; size: UInt32): Prknn_tensor_mem; cdecl;

  // rknn_create_mem_from_fd 函数声明
  Trknn_create_mem_from_fd = function(ctx: rknn_context; fd: Int32; virt_addr: Pointer; size: UInt32; offset: Int32): Prknn_tensor_mem; cdecl;

  // rknn_create_mem_from_mb_blk 函数声明
  Trknn_create_mem_from_mb_blk = function(ctx: rknn_context; mb_blk: Pointer; offset: Int32): Prknn_tensor_mem; cdecl;

  // rknn_create_mem 函数声明
  Trknn_create_mem = function(ctx: rknn_context; size: UInt32): Prknn_tensor_mem; cdecl;

  // rknn_create_mem2 函数声明
  Trknn_create_mem2 = function(ctx: rknn_context; size: UInt64; alloc_flags: UInt64): Prknn_tensor_mem; cdecl;

  // rknn_destroy_mem 函数声明
  Trknn_destroy_mem = function(ctx: rknn_context; mem: Prknn_tensor_mem): Int32; cdecl;

  // rknn_set_weight_mem 函数声明
  Trknn_set_weight_mem = function(ctx: rknn_context; mem: Prknn_tensor_mem): Int32; cdecl;

  // rknn_set_internal_mem 函数声明
  Trknn_set_internal_mem = function(ctx: rknn_context; mem: Prknn_tensor_mem): Int32; cdecl;

  // rknn_set_io_mem 函数声明
  Trknn_set_io_mem = function(ctx: rknn_context; mem: Prknn_tensor_mem; attr: Prknn_tensor_attr): Int32; cdecl;

  // rknn_set_input_shape 函数声明（已弃用）
  Trknn_set_input_shape = function(ctx: rknn_context; attr: Prknn_tensor_attr): Int32; cdecl;

  // rknn_set_input_shapes 函数声明
  Trknn_set_input_shapes = function(ctx: rknn_context; n_inputs: UInt32; attr: Prknn_tensor_attr): Int32; cdecl;

  // rknn_mem_sync 函数声明
  Trknn_mem_sync = function(context: rknn_context; mem: Prknn_tensor_mem; mode: rknn_mem_sync_mode): Int32; cdecl;

const
  RKNN_LIBFILE = 'librknnrt.so';

  OBJ_NAME_MAX_SIZE = 64;
  OBJ_NUMB_MAX_SIZE = 128;

var
  rknn_init: Tknn_init = nil;
  rknn_dup_context: Trknn_dup_context = nil;
  rknn_destroy: Trknn_destroy = nil;
  rknn_query: Trknn_query = nil;
  rknn_inputs_set: Trknn_inputs_set = nil;
  rknn_set_batch_core_num: Trknn_set_batch_core_num = nil;
  rknn_set_core_mask: Trknn_set_core_mask = nil;
  rknn_run: Trknn_run = nil;
  rknn_wait: Trknn_wait = nil;
  rknn_outputs_get: Trknn_outputs_get = nil;
  rknn_outputs_release: Trknn_outputs_release = nil;
  rknn_create_mem_from_phys: Trknn_create_mem_from_phys = nil;
  rknn_create_mem_from_fd: Trknn_create_mem_from_fd = nil;
  rknn_create_mem_from_mb_blk: Trknn_create_mem_from_mb_blk = nil;
  rknn_create_mem: Trknn_create_mem = nil;
  rknn_create_mem2: Trknn_create_mem2 = nil;
  rknn_destroy_mem: Trknn_destroy_mem = nil;
  rknn_set_weight_mem: Trknn_set_weight_mem = nil;
  rknn_set_internal_mem: Trknn_set_internal_mem = nil;
  rknn_set_io_mem: Trknn_set_io_mem = nil;
  rknn_set_input_shape: Trknn_set_input_shape = nil;
  rknn_set_input_shapes: Trknn_set_input_shapes = nil;
  rknn_mem_sync: Trknn_mem_sync = nil;

  function get_rknn_tensor_format_str(const Afmt: rknn_tensor_format): string;
  function get_rknn_tensor_type_str(const AType: rknn_tensor_type): string;
  function get_rknn_tensor_type_len(const AType: rknn_tensor_type): Byte;
  function get_rknn_tensor_qnt_type_str(const AType: rknn_tensor_qnt_type): string;

  function _clip(val, min, max: Single): Integer;
  function clamp(val: Single; min, max: Integer): Integer;
  function qnt_f32_to_affine(f32: Single; zp: Integer; scale: Single): Int8;
  function deqnt_affine_to_f32(qnt: Int8; zp: Int32; scale: Single): Single;
  procedure compute_dfl(tensor: TArray<Single>; dfl_len: Integer; var box: TboxArray);
  function quick_sort_indice_inverse(var input: TArray<Single>; left, right: Integer; var indices: TArray<Integer>): Integer;
  function CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1: Single): Single;
  function nms(validCount: Integer; var outputLocations: TArray<Single>; classIds: TArray<Integer>; var order: TArray<Integer>;
               filterId: Integer; threshold: Single): Integer;
  procedure Softmax(var arr: TArray<Single>);

implementation

{$R-}

procedure Softmax(var arr: TArray<Single>);
var
  maxVal, sum: Single;
  i: Integer;
begin
  if Length(arr) < 1 then exit;
  
  // 找到数组中的最大值
  maxVal := arr[0];
  for i := 1 to High(arr) do
    if arr[i] > maxVal then maxVal := arr[i];

  // 从每个元素中减去最大值以避免溢出
  for i := 0 to High(arr) do
    arr[i] := arr[i] - maxVal;

  // 计算指数并求和
  sum := 0.0;
  for i := 0 to High(arr) do
  begin
    arr[i] := Exp(arr[i]);
    sum := sum + arr[i];
  end;

  // 通过将每个元素除以总和来归一化数组
  for i := 0 to High(arr) do
    arr[i] := arr[i] / sum;
end;

function clamp(val: Single; min, max: Integer): Integer;
begin
  if val > min then
  begin
    if val < max then
      Result := Trunc(val) // 如果 val 在 [min, max] 范围内，返回 val
    else
      Result := max; // 如果 val 大于 max，返回 max
  end
  else
    Result := min; // 如果 val 小于 min，返回 min
end;

function CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1: Single): Single;
var
  w, h, i, u: Single;
begin
  w := Max(0, Min(xmax0, xmax1) - Max(xmin0, xmin1) + 1);
  h := Max(0, Min(ymax0, ymax1) - Max(ymin0, ymin1) + 1);
  i := w * h;
  u := (xmax0 - xmin0 + 1) * (ymax0 - ymin0 + 1) + (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1) - i;
  if u <= 0 then
    Result := 0
  else
    Result := i / u;
end;

function nms(validCount: Integer; var outputLocations: TArray<Single>; classIds: TArray<Integer>; var order: TArray<Integer>;
             filterId: Integer; threshold: Single): Integer;
var
  i, j, n, m: Integer;
  xmin0, ymin0, xmax0, ymax0: Single;
  xmin1, ymin1, xmax1, ymax1: Single;
  iou: Single;
begin
  for i := 0 to validCount - 1 do
  begin
    n := order[i];
    if (n = -1) or (classIds[n] <> filterId) then
      Continue;

    for j := i + 1 to validCount - 1 do
    begin
      m := order[j];
      if (m = -1) or (classIds[m] <> filterId) then
        Continue;

      // 计算第一个框的坐标
      xmin0 := outputLocations[n * 4 + 0];
      ymin0 := outputLocations[n * 4 + 1];
      xmax0 := xmin0 + outputLocations[n * 4 + 2];
      ymax0 := ymin0 + outputLocations[n * 4 + 3];

      // 计算第二个框的坐标
      xmin1 := outputLocations[m * 4 + 0];
      ymin1 := outputLocations[m * 4 + 1];
      xmax1 := xmin1 + outputLocations[m * 4 + 2];
      ymax1 := ymin1 + outputLocations[m * 4 + 3];

      // 计算 IoU
      iou := CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      // 如果 IoU 大于阈值，则抑制该框
      if iou > threshold then
        order[j] := -1;
    end;
  end;
  Result := 0;
end;

function quick_sort_indice_inverse(var input: TArray<Single>; left, right: Integer; var indices: TArray<Integer>): Integer;
var
  key: Single;
  key_index, low, high: Integer;
begin
  // 检查 left 和 right 的值是否有效
  low := left;
  high := right;
  if left < right then
  begin
    key_index := indices[left];
    key := input[left];
    while (low < high) do
    begin
      while (low < high) and (input[high] <= key) do Dec(high);

      input[low] := input[high];
      indices[low] := indices[high];

      while (low < high) and (input[low] >= key) do Inc(low);

      input[high] := input[low];
      indices[high] := indices[low];
    end;

    input[low] := key;
    indices[low] := key_index;

    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  end;
  Result := low;
end;

procedure compute_dfl(tensor: TArray<Single>; dfl_len: Integer; var box: TboxArray);
var
  b, i: Integer;
  exp_t: TArray<Single>;
  exp_sum, acc_sum: Single;
begin
  SetLength(exp_t, dfl_len); // 初始化 exp_t 数组
  for b := 0 to 3 do
  begin
    exp_sum := 0;
    acc_sum := 0;

    // 计算 exp_t 和 exp_sum
    for i := 0 to dfl_len - 1 do
    begin
      exp_t[i] := Exp(tensor[i + b * dfl_len]);
      exp_sum := exp_sum + exp_t[i];
    end;

    // 计算 acc_sum
    for i := 0 to dfl_len - 1 do
    begin
      acc_sum := acc_sum + (exp_t[i] / exp_sum) * i;
    end;

    // 将结果存储到 box 中
    box[b] := acc_sum;
  end;
end;

function deqnt_affine_to_f32(qnt: Int8; zp: Int32; scale: Single): Single;
begin
  Result := (Single(qnt) - Single(zp)) * scale;
end;

function _clip(val, min, max: Single): Integer;
begin
  if val <= min then
    Result := Trunc(min)
  else if val >= max then
    Result := Trunc(max)
  else
    Result := Trunc(val);
end;

function qnt_f32_to_affine(f32: Single; zp: Integer; scale: Single): Int8;
var
  dst_val: Single;
begin
  dst_val := (f32 / scale) + zp;
  Result := _clip(dst_val, -128, 127);
end;

function get_rknn_tensor_type_len(const AType: rknn_tensor_type): Byte;
begin
  case Atype of
    RKNN_TENSOR_FLOAT32: Result:=4;
    RKNN_TENSOR_FLOAT16: Result:=2;
    RKNN_TENSOR_INT8: Result:=1;
    RKNN_TENSOR_UINT8: Result:=1;
    RKNN_TENSOR_INT16: Result:=2;
    RKNN_TENSOR_UINT16: Result:=2;
    RKNN_TENSOR_INT32: Result:=4;
    RKNN_TENSOR_UINT32: Result:=4;
    RKNN_TENSOR_INT64: Result:=8;
    RKNN_TENSOR_BOOL: Result:=1;
  else
    Result:=0;
  end;
end;

function get_rknn_tensor_qnt_type_str(const AType: rknn_tensor_qnt_type): string;
begin
  case Atype of
    RKNN_TENSOR_QNT_NONE: Result:='NONE';
    RKNN_TENSOR_QNT_DFP: Result:='DFP';
    RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC: Result:='AFFINE';
  else
    Result:='UNKNOW';
  end;
end;

function get_rknn_tensor_type_str(const AType: rknn_tensor_type): string;
begin
  case Atype of
    RKNN_TENSOR_FLOAT32: Result:='FP32';
    RKNN_TENSOR_FLOAT16: Result:='FP16';
    RKNN_TENSOR_INT8: Result:='INT8';
    RKNN_TENSOR_UINT8: Result:='UINT8';
    RKNN_TENSOR_INT16: Result:='INT16';
    RKNN_TENSOR_UINT16: Result:='UINT16';
    RKNN_TENSOR_INT32: Result:='INT32';
    RKNN_TENSOR_UINT32: Result:='UINT32';
    RKNN_TENSOR_INT64: Result:='INT64';
    RKNN_TENSOR_BOOL: Result:='BOOL';
  else
    Result:='UNKNOW';
  end;
end;

function get_rknn_tensor_format_str(const Afmt: rknn_tensor_format): string;
begin
  case Afmt of
    RKNN_TENSOR_NCHW: Result:='NCHW';
    RKNN_TENSOR_NHWC: Result:='NHWC';
    RKNN_TENSOR_NC1HWC2: Result:='NC1HWC2';
    RKNN_TENSOR_UNDEFINED: Result:='UNDEFINED';
  else
    Result:='UNKNOW';
  end;
end;

initialization

end.