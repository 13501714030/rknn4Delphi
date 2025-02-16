{**************************************************************************

rknn4Delphi (Rockchip RKNN Inference Frame Delphi Wrapped)

author: Tom Yea
create time: 2025/1/25
contact: tom_ye@qq.com

本单元为rknn基础类，基础类提供了一些最基础的操作，如：rknn库调用，初始化，
模型文件的输入输出信息提取，推理的前处理和调用推理等操作

****************************************************************************}

unit rknnBase;

interface

uses
  System.SysUtils, System.Types, System.IOUtils, System.Classes,
  FMX.Memo,
  FMX.Objects,
  FMX.Dialogs,
  FMX.Graphics,
  System.UITypes,
  FMX.Utils,
  FMX.Types,
  System.Math,
  System.Generics.Defaults,
  System.Generics.Collections,
  rknn_api;

type

  //画框颜色指针，由于画框是实时进行，所以需要用指针速度比直接赋值快
  PClassColors = ^TClassColors;
  TClassColors = array of TAlphaColor;

  //画框和其它信息等数据
  TDrawType = (dtFilled,dtFrame);

  TRkPlatform = (rk3588, rk3568, rk3566, rv1126, rv1106);

  TrknnBase = class(TObject)
  private
    FLib: HMODULE;
    FInputImgWidth: Word;
    FInputImgHeight: Word;

    function OpenModelFile(const AFilename: string; var AModelSize: Integer; const ANeedDecrypt: Boolean = false): TBytes;
    procedure SetClassColors;

  protected
    FCtx: rknn_context;
    FIsQuantized: Boolean; // 是否为量化模型
    FIOCount: rknn_input_output_num;
    FRknnVerInfo: rknn_sdk_version;
    FRknnAPIVer: ShortString;
    FRknnDRVVer: ShortString;
    FInputTensorsAttr: array of rknn_tensor_attr;
    FOutputTensorsAttr: array of rknn_tensor_attr;
    FModelFile: string;
    FPlatform: TRkPlatform;
    FInputTensorType: rknn_tensor_type;
    FOutputs: rknn_outputs;
    FOutputFloat: Boolean;

    FOrgBitmap: TBitmap;                               // 输入的原始图像
    FResizedBitmap: TBitmap;                           // 送入网络变化过尺寸的图片
    FOrgBMPWidth: Word;                                // 原始图宽度
    FOrgBMPHeight: Word;                               // 原词图高度
    FImgW_rate, FImgH_rate: Single;                    // 原始图和变化过尺寸后的尺寸比例，画框时需要使用
    FResizedRect: TRect;                               // 最终缩放尺寸后的Rect
    FClassColors: TClassColors;                        // 分类颜色数组
    FClassCount: Integer;                              // 类别数量
    FDrawType: TDrawType;                              // 画的框类型
    FConfThres: Single;                                // 置信度
    FNMSThres: Single;                                 // 非极大化抑制阈值

    FClassNames: TStrings;                             // 类别
    FExcludeClassIDs: TStrings;                             // 排除类别ID

    function LoadLibrary(const ALibFile: string; var AErrMsg: string): Boolean; virtual;

    function GetRknnAPIVer: string;
    function GetRknnDRVVer: string;

    // 图片按比例缩小，通用函数，后面子类都可以调用
    procedure CreateResizedBMP(ABMP: TBitmap);

    procedure DoPreprocess; virtual;  // 预处理 之类可以继承并实现额外的处理
    procedure DoPostprocess; virtual; // 后处理
    procedure ClearOutputsBuffer;     // 释放输出申请的内存
    procedure Detect(ABMP: TBitmap); virtual;  //识别

    //把结果数据画在bitmap上
    procedure DrawResult(ABitmap: TBitmap; const AInfo: string; const ADrawLabel: Boolean = true);

  public
//    LogMemo: TMemo;
//    Img: TImage;

    constructor Create;
    destructor Destroy; override;

    function LoadModelFile(const AFilename: string; const ANeedDecrypt: Boolean = false): Boolean; virtual;

    function GetTensorsInfo: string;

    procedure LoadClassNames(const AClassFile: string); overload;
    procedure LoadClassNames(const AData: array of string); overload;
    procedure LoadClassNames(const AList: TStrings); overload;

    procedure ExcludeClassIDs(const ACExcludeClassFile: string); overload;
    procedure ExcludeClassIDs(const AData: array of Word); overload;
    procedure ExcludeClassIDs(const AList: TStrings); overload;

  published
    property RknnApiVer: string read GetRknnAPIVer;
    property RknnDrvVer: string read GetRknnDRVVer;
    property Quantized: Boolean read FIsQuantized;
    property OutputFloat: Boolean read FOutputFloat write FOutputFloat;
    property InputTensorType: rknn_tensor_type read FInputTensorType write FInputTensorType;
    property InputImgWidth: Word read FInputImgWidth;
    property InputImgHeight: Word read FInputImgHeight;
    property RKPlatform: TRkPlatform read FPlatform write FPlatform;
    property ClassCount: Integer read FClassCount;
    property ModelFile: string read FModelFile;
    property DrawType: TDrawType read FDrawType write FDrawType;
    property ConfThres: Single read FConfThres write FConfThres;
    property NMSThres: Single read FNMSThres write FNMSThres;
  end;

  //工具类 ===================================================================================

  TArrayHelper = class
  public
    class procedure SortArray<T>(var AData: TArray<T>; const AKeepCount: Integer; const AIsAscending: Boolean); //数组排序
    class function CreateSortArray<T>(const ASrc: TArray<T>; const AKeepCount: Integer; const AIsAscending: Boolean): TArray<T>;
    class procedure RemoveDuplicates<T>(var AData: TArray<T>); //数组去重
    class function CreateRemoveDuplicates<T>(var ASrc: TArray<T>): TArray<T>;

  end;

  TPicColorMode = (cmRGB, cmBGR);
  function BMPToByteArray(ABMP: TBitmap; const AMode: TPicColorMode = cmBGR): TArray<Byte>;
  procedure ByteArrayToBMP(ABMP: TBitmap; AData: TArray<Byte>; const AWidth, AHeight: Integer; const AOpacity: Single; const AMode: TPicColorMode = cmBGR);

var
  TRectColors: array [0..21] of TAlphaColor = (TAlphaColorRec.Lime,TAlphaColorRec.Wheat,TAlphaColorRec.Yellow,TAlphaColorRec.Violet,
   TAlphaColorRec.Turquoise,TAlphaColorRec.Tomato,TAlphaColorRec.Thistle,TAlphaColorRec.Steelblue,TAlphaColorRec.Skyblue,TAlphaColorRec.Seagreen,
   TAlphaColorRec.Sandybrown,TAlphaColorRec.Salmon,TAlphaColorRec.Royalblue,TAlphaColorRec.Powderblue,TAlphaColorRec.Plum,
   TAlphaColorRec.Pink,TAlphaColorRec.Blueviolet,TAlphaColorRec.Burlywood,TAlphaColorRec.Coral,TAlphaColorRec.Cornflowerblue,
   TAlphaColorRec.Greenyellow,TAlphaColorRec.Gold);

implementation

{$R-}

uses MiscFmx;

class procedure TArrayHelper.RemoveDuplicates<T>(var AData: TArray<T>);
var
  UniqueSet: TDictionary<T, Integer>;
  i: Integer;
  UniqueList: TList<T>;
begin
  // 使用 TDictionary 去重
  UniqueSet := TDictionary<T, Integer>.Create;
  UniqueList := TList<T>.Create;
  try
    for i := 0 to High(AData) do
    begin
      if not UniqueSet.ContainsKey(AData[i]) then
      begin
        UniqueSet.Add(AData[i], 0); // 0 是占位值，实际未使用
        UniqueList.Add(AData[i]);   // 将唯一元素添加到列表中
      end;
    end;

    // 将去重后的结果赋值回 AData
    AData := UniqueList.ToArray;
  finally
    UniqueSet.Free;
    UniqueList.Free;
  end;
end;

class function TArrayHelper.CreateRemoveDuplicates<T>(var ASrc: TArray<T>): TArray<T>;
begin
  Result:=Copy(ASrc);
  TArrayHelper.RemoveDuplicates<T>(Result);
end;

class function TArrayHelper.CreateSortArray<T>(const ASrc: TArray<T>; const AKeepCount: Integer; const AIsAscending: Boolean): TArray<T>;
begin
  Result:=Copy(ASrc);
  TArrayHelper.SortArray<T>(Result,AKeepCount,AIsAscending);
end;

class procedure TArrayHelper.SortArray<T>(var AData: TArray<T>;
  const AKeepCount: Integer; const AIsAscending: Boolean);
var
  Comparer: IComparer<T>;
  DataLength: Integer;
begin
  // 获取数组长度
  DataLength := Length(AData);
  // 根据排序方式选择比较器
  if AIsAscending then
    Comparer := TComparer<T>.Default
  else
    Comparer := TComparer<T>.Construct(
      function(const Left, Right: T): Integer
      begin
        Result := -TComparer<T>.Default.Compare(Left, Right); // 降序
      end);

  // 使用快速排序算法进行排序
  TArray.Sort<T>(AData, Comparer);

  // 如果 AKeepCount 大于数组长度或者小于1，则调整为数组长度
  if (AKeepCount > DataLength) or (AKeepCount < 1) then exit;

  // 保留前 AKeepCount 个元素
  SetLength(AData, AKeepCount);
end;

procedure ByteArrayToBMP(ABMP: TBitmap; AData: TArray<Byte>; const AWidth, AHeight: Integer; const AOpacity: Single; const AMode: TPicColorMode = cmBGR);
var
  i, j, Index: Integer;
  PColors: PAlphaColorArray;
  BitmapData: TBitmapData;
  R, G, B, A: Byte;
begin
  if not Assigned(ABMP) or (Length(AData) = 0) then
    Exit;

  A := Round(AOpacity * 255);
  Index := 0;
  ABMP.SetSize(AWidth, AHeight);
  ABMP.Map(TMapAccess.Write, BitmapData);

  try
    if AMode = cmRGB then
      for i := 0 to ABMP.Height - 1 do
      begin
        PColors := PAlphaColorArray(BitmapData.GetScanline(i));
        for j := 0 to ABMP.Width - 1 do
        begin
          R := AData[Index];
          G := AData[Index + 1];
          B := AData[Index + 2];
          TAlphaColorRec(PColors[j]).R := R;
          TAlphaColorRec(PColors[j]).G := G;
          TAlphaColorRec(PColors[j]).B := B;
          TAlphaColorRec(PColors[j]).A := A;
          Inc(Index, 3);
        end;
      end
    else
      for i := 0 to ABMP.Height - 1 do
      begin
        PColors := PAlphaColorArray(BitmapData.GetScanline(i));
        for j := 0 to ABMP.Width - 1 do
        begin
          B := AData[Index];
          G := AData[Index + 1];
          R := AData[Index + 2];
          TAlphaColorRec(PColors[j]).B := B;
          TAlphaColorRec(PColors[j]).G := G;
          TAlphaColorRec(PColors[j]).R := R;
          TAlphaColorRec(PColors[j]).A := A;
          Inc(Index, 3);
        end;
      end;
  finally
    ABMP.Unmap(BitmapData);
  end;
end;

function BMPToByteArray(ABMP: TBitmap; const AMode: TPicColorMode = cmBGR)
  : TArray<Byte>;
var
  i, j, Index: Integer;
  PColors: PAlphaColorArray;
  BitmapData: TBitmapData;
  R, G, B: Byte;
begin
  Result := nil;
  if not Assigned(ABMP) then
    Exit;
  SetLength(Result, ABMP.Width * ABMP.Height * 3);
  Index := 0;
  ABMP.Map(TMapAccess.ReadWrite, BitmapData);
  try
    if AMode = cmRGB then
      for i := 0 to ABMP.Height - 1 do
      begin
        PColors := PAlphaColorArray(BitmapData.GetScanline(i));
        for j := 0 to ABMP.Width - 1 do
        begin
          R := TAlphaColorRec(PColors[j]).R;
          G := TAlphaColorRec(PColors[j]).G;
          B := TAlphaColorRec(PColors[j]).B;
          Result[Index] := R;
          Result[Index + 1] := G;
          Result[Index + 2] := B;
          Inc(Index, 3);
        end;
      end
    else
      for i := 0 to ABMP.Height - 1 do
      begin
        PColors := PAlphaColorArray(BitmapData.GetScanline(i));
        for j := 0 to ABMP.Width - 1 do
        begin
          R := TAlphaColorRec(PColors[j]).R;
          G := TAlphaColorRec(PColors[j]).G;
          B := TAlphaColorRec(PColors[j]).B;
          Result[Index] := B;
          Result[Index + 1] := G;
          Result[Index + 2] := R;
          Inc(Index, 3);
        end;
      end;
  finally
    ABMP.Unmap(BitmapData);
  end;
end;

{ TRknnBase }

procedure TRknnBase.ClearOutputsBuffer;
var
  ret: Integer;
begin
  if Length(FOutputs) < 1 then exit;
  // 释放输出缓存区
  ret := rknn_outputs_release(FCtx, FIOCount.n_output, @FOutputs[0]);
  if ret <> RKNN_SUCC then
    raise Exception.Create('error on rknn_outputs_release() ! code: ' + ret.ToString);
end;

constructor TRknnBase.Create;
var
  s: string;
begin
  inherited Create;
  FResizedBitmap:=TBitmap.Create(640, 640);
  FClassNames:=TStringList.Create;
  FExcludeClassIDs:=TStringList.Create;
  FNMSThres:=0.5;
  FConfThres:=0.5;
  FInputTensorType := RKNN_TENSOR_UINT8;  //默认输入图像数据转成Byte类型
  FDrawType:=TDrawType.dtFrame;           //默认画框不填充
  if not LoadLibrary(System.IOUtils.TPath.Combine(System.IOUtils.TPath.GetHomePath, RKNN_LIBFILE), s) then
    raise Exception.Create(s);
end;

destructor TRknnBase.Destroy;
begin
  FreeAndNil(FExcludeClassIDs);
  FreeAndNil(FClassNames);
  FreeAndNil(FResizedBitmap);
  ClearOutputsBuffer;
  if FCtx > 0 then rknn_destroy(FCtx);
  if FLib <> 0 then FreeLibrary(FLib);
  inherited Destroy;
end;

procedure TRknnBase.Detect(ABMP: TBitmap);
var
  ret: Integer;
begin
  if not Assigned(ABMP) then Exit;
  CreateResizedBMP(ABMP); // 把输入的图像缩到模型输入的规定尺寸
  DoPreprocess; // 然后做预处理
  // 进行推理
  ret := rknn_run(FCtx, nil);
  if ret <> RKNN_SUCC then
    raise Exception.Create('error on rknn_run() ! code: ' + ret.ToString);
end;

procedure TRknnBase.DoPostprocess;
var
  i, ret: Integer;
begin
  Self.ClearOutputsBuffer;
  FOutputs:=nil;
  SetLength(FOutputs, FIOCount.n_output); // 给输出分配空间
  for i := 0 to High(FOutputs) do
  begin
    FOutputs[i].index:=i;
    FOutputs[i].want_float := Integer(FOutputFloat);  //输出是否要求是浮点型
    FOutputs[i].is_prealloc:= 0;                      //是否要分配缓冲区空间
  end;

  // 从rknn获取输出数据到 Outputs
  ret := rknn_outputs_get(FCtx, FIOCount.n_output, @FOutputs[0], nil);
  if ret <> RKNN_SUCC then
    raise Exception.Create('error on rknn_outputs_get() ! code: ' +
      ret.ToString);

//  // 把结果数据填充到原始数据数组中
//  FOrgOutputs := nil;
//  SetLength(FOrgOutputs, FIOCount.n_output);
//  for i := 0 to FIOCount.n_output - 1 do
//  begin
//    FOrgOutputs[i].Index := i;
//    FOrgOutputs[i].DataSize := FOutputTensorsAttr[i].size;//  Outputs[i].size div SizeOf(Single);
//    // 输出的buffer数据大小，需要除以占用的字节数后才是数组的总大小 弃用，直接从 Tensor Attr中获取，不需要除以类型占用的字节数
//    SetLength(FOrgOutputs[i].Data, FOrgOutputs[i].DataSize);
//    Move(FOutputs[i].buf^, FOrgOutputs[i].Data[0], FOutputs[i].size); //注意这里需要使用output的size，因为它是加上了类型字节
//  end;

//  if ANeedDecode then  //如果需要解码的
//  begin
//    YoloV8Decode(Outputs);
//  end;

end;

procedure TRknnBase.DoPreprocess;
var
  Input: rknn_input;
  ImgData: TBytes;
  ret: Integer;

begin
  ImgData := BMPToByteArray(FResizedBitmap);

  Input.Index := 0;
  Input.type_ := Integer(FInputTensorType); //Integer(RKNN_TENSOR_UINT8); //输入类型，一般都是0~255的rgb数组 //Integer(FInputTensorType);
  Input.size := FResizedBitmap.Width * Self.FResizedBitmap.Height * 3 *
                get_rknn_tensor_type_len(rknn_tensor_type(FInputTensorsAttr[0].type_));
  Input.fmt := FInputTensorsAttr[0].fmt;
  Input.buf := @ImgData[0];

  ret := rknn_inputs_set(FCtx, FIOCount.n_input, @Input);
  if ret <> RKNN_SUCC then
    raise Exception.Create('error on rknn_inputs_set() ! code: ' + ret.ToString);
end;

procedure TRknnBase.DrawResult(ABitmap: TBitmap; const AInfo: string; const ADrawLabel: Boolean = true);
begin

end;

procedure TrknnBase.ExcludeClassIDs(const ACExcludeClassFile: string);
begin
  if not FileExists(ACExcludeClassFile) then raise Exception.Create('class file '+ACExcludeClassFile+' does not exist !');
  try
    FExcludeClassIDs.LoadFromFile(ACExcludeClassFile, TEncoding.UTF8);
  except
    FExcludeClassIDs.LoadFromFile(ACExcludeClassFile);
  end;
end;

procedure TrknnBase.ExcludeClassIDs(const AList: TStrings);
begin
  FExcludeClassIDs.Assign(AList);
end;

procedure TrknnBase.ExcludeClassIDs(const AData: array of Word);
var
  i: Integer;
begin
  FExcludeClassIDs.Clear;
  for i := 0 to High(AData) do FExcludeClassIDs.Add(AData[i].ToString);
end;

function TrknnBase.GetTensorsInfo: string;
var
  AList: TStrings;
  i: Integer;
begin
  AList:=TStringList.Create;
  try
    AList.Add('rockchip rknn API verion: '+GetRknnAPIVer);
    AList.Add('    model file: '+FModelFile);
    AList.Add(Format('    current model input num: %d; output num: %d',[FIOCount.n_input, FIOCount.n_output]));
    AList.Add('');
    AList.Add('input tensor:');
    for i:=0 to High(FInputTensorsAttr) do
      AList.Add(Format('    index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%.6f',
                      [FInputTensorsAttr[i].Index, FInputTensorsAttr[i].name,
                       FInputTensorsAttr[i].n_dims, FInputTensorsAttr[i].dims[0],
                       FInputTensorsAttr[i].dims[1], FInputTensorsAttr[i].dims[2],
                       FInputTensorsAttr[i].dims[3], FInputTensorsAttr[i].n_elems,
                       FInputTensorsAttr[i].size, get_rknn_tensor_format_str(rknn_tensor_format(FInputTensorsAttr[i].fmt)),
                       get_rknn_tensor_type_str(rknn_tensor_type(FInputTensorsAttr[i].type_)),
                       get_rknn_tensor_qnt_type_str(rknn_tensor_qnt_type(FInputTensorsAttr[i].qnt_type)), FInputTensorsAttr[i].zp,
                       FInputTensorsAttr[i].scale]) + sLineBreak);
    AList.Add('');
    AList.Add('output tensor:');
    for i:=0 to High(FOutputTensorsAttr) do
      AList.Add(Format('    index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%.6f',
                      [FOutputTensorsAttr[i].Index, FOutputTensorsAttr[i].name,
                       FOutputTensorsAttr[i].n_dims, FOutputTensorsAttr[i].dims[0],
                       FOutputTensorsAttr[i].dims[1], FOutputTensorsAttr[i].dims[2],
                       FOutputTensorsAttr[i].dims[3], FOutputTensorsAttr[i].n_elems,
                       FOutputTensorsAttr[i].size, get_rknn_tensor_format_str(rknn_tensor_format(FOutputTensorsAttr[i].fmt)),
                       get_rknn_tensor_type_str(rknn_tensor_type(FOutputTensorsAttr[i].type_)),
                       get_rknn_tensor_qnt_type_str(rknn_tensor_qnt_type(FOutputTensorsAttr[i].qnt_type)), FOutputTensorsAttr[i].zp,
                       FOutputTensorsAttr[i].scale]) + sLineBreak);
  finally
    Result:=AList.Text;
    FreeAndNil(AList);
  end;
end;

function TRknnBase.GetRknnAPIVer: string;
begin
  Result:= FRknnAPIVer;
end;

function TRknnBase.GetRknnDRVVer: string;
begin
  Result:= FRknnDRVVer;
end;

procedure TRknnBase.CreateResizedBMP(ABMP: TBitmap);
var
  ImgRate, ImgW, ImgH: Single;
begin
  FOrgBMPWidth := ABMP.Width;
  FOrgBMPHeight := ABMP.Height;
  ImgRate := ABMP.Width / ABMP.Height;

  FResizedBitmap.SetSize(FInputImgWidth, FInputImgHeight);

  if FOrgBMPWidth >= FOrgBMPHeight then // 判断是否是竖向或横向
  begin
    FImgW_rate := ABMP.Width / FInputImgWidth;
    ImgW := FInputImgWidth;
    ImgH := FInputImgWidth / ImgRate;
    FImgH_rate := ABMP.Height / ImgH;
  end else begin
    FImgH_rate := ABMP.Height / FInputImgHeight;
    ImgW := FInputImgHeight * ImgRate;
    ImgH := FInputImgHeight;
    FImgW_rate := ABMP.Width / ImgW;
  end;

  FResizedRect.Left := 0;
  FResizedRect.Right := 0;
  FResizedRect.Width := Round(ImgW);
  FResizedRect.Height := Round(ImgH);

  FResizedBitmap.Canvas.BeginScene();
  try
    FResizedBitmap.Canvas.Fill.Color := TAlphaColorRec.Gray;
    FResizedBitmap.Canvas.FillRect(Rect(0, 0, FInputImgWidth, FInputImgHeight), 1);
    FResizedBitmap.Canvas.DrawBitmap(ABMP, RectF(0, 0, ABMP.Width, ABMP.Height), RectF(0, 0, ImgW, ImgH), 1, true);
  finally
    FResizedBitmap.Canvas.EndScene;
//    if Assigned(Img) then Img.Bitmap.Assign(FResizedBitmap);
  end;
end;

procedure TRknnBase.LoadClassNames(const AClassFile: string);
begin
  if not FileExists(AClassFile) then raise Exception.Create('class file '+AClassFile+' does not exist !');
  try
    FClassNames.LoadFromFile(AClassFile, TEncoding.UTF8);
    SetClassColors;
  except
    FClassNames.LoadFromFile(AClassFile);
    SetClassColors;
  end;
end;

procedure TRknnBase.LoadClassNames(const AData: array of string);
var
  i: Integer;
begin
  FClassNames.Clear;
  for i := 0 to High(AData) do FClassNames.Add(AData[i]);
  SetClassColors;
end;

procedure TRknnBase.LoadClassNames(const AList: TStrings);
begin
  FClassNames.Assign(AList);
  SetClassColors;
end;

function TRknnBase.LoadLibrary(const ALibFile: string; var AErrMsg: string): Boolean;
begin
  Result := false;
  if not FileExists(ALibFile) then
  begin
    AErrMsg := 'file ' + ALibFile + ' does not exist !';
    Exit;
  end;

  FLib := System.SysUtils.LoadLibrary(PWideChar(ALibFile));
  if FLib = 0 then
  begin
    AErrMsg := 'load ' + ALibFile + ' error !';
    Exit;
  end;

  rknn_init := GetProcAddress(FLib, 'rknn_init');
  if not Assigned(rknn_init) then
  begin
    AErrMsg := 'load process address at rknn_init() error !';
    Exit;
  end;

  rknn_dup_context := GetProcAddress(FLib, 'rknn_dup_context');
  if not Assigned(rknn_dup_context) then
  begin
    AErrMsg := 'load process address at rknn_dup_context() error !';
    Exit;
  end;

  rknn_destroy := GetProcAddress(FLib, 'rknn_destroy');
  if not Assigned(rknn_destroy) then
  begin
    AErrMsg := 'load process address at rknn_destroy() error !';
    Exit;
  end;

  rknn_query := GetProcAddress(FLib, 'rknn_query');
  if not Assigned(rknn_query) then
  begin
    AErrMsg := 'load process address at rknn_query() error !';
    Exit;
  end;

  rknn_inputs_set := GetProcAddress(FLib, 'rknn_inputs_set');
  if not Assigned(rknn_inputs_set) then
  begin
    AErrMsg := 'load process address at rknn_inputs_set() error !';
    Exit;
  end;

  rknn_set_batch_core_num := GetProcAddress(FLib, 'rknn_set_batch_core_num');
  if not Assigned(rknn_set_batch_core_num) then
  begin
    AErrMsg := 'load process address at rknn_set_batch_core_num() error !';
    Exit;
  end;

  rknn_set_core_mask := GetProcAddress(FLib, 'rknn_set_core_mask');
  if not Assigned(rknn_set_core_mask) then
  begin
    AErrMsg := 'load process address at rknn_set_core_mask() error !';
    Exit;
  end;

  rknn_run := GetProcAddress(FLib, 'rknn_run');
  if not Assigned(rknn_run) then
  begin
    AErrMsg := 'load process address at rknn_run() error !';
    Exit;
  end;

  rknn_wait := GetProcAddress(FLib, 'rknn_wait');
  if not Assigned(rknn_wait) then
  begin
    AErrMsg := 'load rknn_wait() error !';
    Exit;
  end;

  rknn_outputs_get := GetProcAddress(FLib, 'rknn_outputs_get');
  if not Assigned(rknn_outputs_get) then
  begin
    AErrMsg := 'load rknn_outputs_get() error !';
    Exit;
  end;

  rknn_outputs_release := GetProcAddress(FLib, 'rknn_outputs_release');
  if not Assigned(rknn_outputs_release) then
  begin
    AErrMsg := 'load rknn_outputs_release() error !';
    Exit;
  end;

  rknn_create_mem_from_phys := GetProcAddress(FLib,
    'rknn_create_mem_from_phys');
  if not Assigned(rknn_create_mem_from_phys) then
  begin
    AErrMsg := 'load rknn_create_mem_from_phys() error !';
    Exit;
  end;

  rknn_create_mem_from_fd := GetProcAddress(FLib, 'rknn_create_mem_from_fd');
  if not Assigned(rknn_create_mem_from_fd) then
  begin
    AErrMsg := 'load rknn_create_mem_from_fd() error !';
    Exit;
  end;

  rknn_create_mem := GetProcAddress(FLib, 'rknn_create_mem');
  if not Assigned(rknn_create_mem) then
  begin
    AErrMsg := 'load process address at rknn_create_mem() error !';
    Exit;
  end;

  rknn_create_mem2 := GetProcAddress(FLib, 'rknn_create_mem2');
  if not Assigned(rknn_create_mem2) then
  begin
    AErrMsg := 'load process address at rknn_create_mem2() error !';
    Exit;
  end;

  rknn_destroy_mem := GetProcAddress(FLib, 'rknn_destroy_mem');
  if not Assigned(rknn_destroy_mem) then
  begin
    AErrMsg := 'load process address at rknn_destroy_mem() error !';
    Exit;
  end;

  rknn_set_weight_mem := GetProcAddress(FLib, 'rknn_set_weight_mem');
  if not Assigned(rknn_set_weight_mem) then
  begin
    AErrMsg := 'load process address at rknn_set_weight_mem() error !';
    Exit;
  end;

  rknn_set_internal_mem := GetProcAddress(FLib, 'rknn_set_internal_mem');
  if not Assigned(rknn_set_internal_mem) then
  begin
    AErrMsg := 'load process address at rknn_set_internal_mem() error !';
    Exit;
  end;

  rknn_set_io_mem := GetProcAddress(FLib, 'rknn_set_io_mem');
  if not Assigned(rknn_set_io_mem) then
  begin
    AErrMsg := 'load process address at rknn_set_io_mem() error !';
    Exit;
  end;

  rknn_set_input_shape := GetProcAddress(FLib, 'rknn_set_input_shape');
  if not Assigned(rknn_set_input_shape) then
  begin
    AErrMsg := 'load process address at rknn_set_input_shape() error !';
    Exit;
  end;

  rknn_set_input_shapes := GetProcAddress(FLib, 'rknn_set_input_shapes');
  if not Assigned(rknn_set_input_shapes) then
  begin
    AErrMsg := 'load process address at rknn_set_input_shapes() error !';
    Exit;
  end;

  rknn_mem_sync := GetProcAddress(FLib, 'rknn_mem_sync');
  if not Assigned(rknn_mem_sync) then
  begin
    AErrMsg := 'load process address at rknn_mem_sync() error !';
    Exit;
  end;

  Result := true;
end;

function TRknnBase.LoadModelFile(const AFilename: string; const ANeedDecrypt: Boolean = false): Boolean;
var
  Data: TBytes;
  ModelSize: Integer;
  i, ret: Integer;
begin
  Result:=false;
  if not FileExists(AFileName) then raise Exception.Create('model file '+AFileName+' does not exist !');

  FModelFile:=AFileName;
  FInputTensorsAttr:=nil;
  FOutputTensorsAttr:=nil;
  Data := OpenModelFile(AFilename, ModelSize, ANeedDecrypt);
  ret := rknn_init(@FCtx, @Data[0], ModelSize, 0, nil);
  if ret <> RKNN_SUCC then
    raise Exception.Create('open and init model file error ! return code: ' + ret.ToString);

  //查询当前rknn sdk版本
  ret := rknn_query(FCtx, RKNN_QUERY_SDK_VERSION, @FRknnVerInfo, SizeOf(FRknnVerInfo));
  if ret <> RKNN_SUCC then
    raise Exception.Create('query RKNN_QUERY_SDK_VERSION error ! return code: ' + ret.ToString);
  FRknnAPIVer:=ShortString(FRknnVerInfo.api_version);
  FRknnDRVVer:=ShortString(FRknnVerInfo.drv_version);

  // 查询输入输出数量
  ret := rknn_query(FCtx, RKNN_QUERY_IN_OUT_NUM, @FIOCount, SizeOf(FIOCount));
  if ret <> RKNN_SUCC then
    raise Exception.Create('query RKNN_QUERY_IN_OUT_NUM error ! return code: ' + ret.ToString);

  // 查询输入属性
  SetLength(FInputTensorsAttr, FIOCount.n_input); // 根据前面的查询设置InputTensors数量
  // FillChar(FInputTensors, FIOCount.n_input * SizeOf(rknn_tensor_attr), 0);
  // 遍历模型所有输入（网络可能有多个输入，这里为了兼容多输入，使用for循环遍历）
  for i := 0 to FIOCount.n_input - 1 do
  begin
    FInputTensorsAttr[i].Index := i;
    // 使用rknn_query函数获取模型输入信息，存储在input_attrs
    ret := rknn_query(FCtx, RKNN_QUERY_INPUT_ATTR, @FInputTensorsAttr[i], SizeOf(rknn_tensor_attr));
    if ret <> RKNN_SUCC then
      raise Exception.Create('query RKNN_QUERY_INPUT_ATTR at input tensor ' + i.ToString + ' error ! code: ' + ret.ToString);
  end;
  FIsQuantized:=FInputTensorsAttr[0].type_ = 2;  //是否为量化模型，2=int8; 3=uint8
  if FInputTensorsAttr[0].fmt = Integer(RKNN_TENSOR_NCHW) then
  begin
    FInputImgWidth := FInputTensorsAttr[0].dims[3];
    FInputImgHeight := FInputTensorsAttr[0].dims[2];
  end else begin //RKNN_TENSOR_NHWC
    FInputImgWidth := FInputTensorsAttr[0].dims[2];
    FInputImgHeight := FInputTensorsAttr[0].dims[1];
  end;

  // 查询输出属性
  SetLength(FOutputTensorsAttr, FIOCount.n_output); // 根据前面的查询设置InputTensors数量

  for i := 0 to FIOCount.n_output - 1 do
  begin
    FOutputTensorsAttr[i].Index := i;
    // 使用rknn_query函数获取模型输入信息，存储在input_attrs
    ret := rknn_query(FCtx, RKNN_QUERY_OUTPUT_ATTR, @FOutputTensorsAttr[i], SizeOf(rknn_tensor_attr));
    if ret <> RKNN_SUCC then
      raise Exception.Create('query RKNN_QUERY_OUTPUT_ATTR at output tensor ' + i.ToString + ' error ! code: ' + ret.ToString);
  end;
  Result:=true;
end;

function TRknnBase.OpenModelFile(const AFilename: string; var AModelSize: Integer; const ANeedDecrypt: Boolean = false): TBytes;
  function FillData(AStream: TStream): TBytes;
  begin
    Result:=nil;
    // 获取文件大小
    AModelSize := AStream.size;
    // 分配内存并读取文件内容
    SetLength(Result, AModelSize); // 动态数组分配内存
    AStream.Position:=0;
    AStream.ReadBuffer(Result[0], AModelSize); // 读取文件内容到动态数组
  end;
var
  FileStream: TStream;
begin
  Result := nil; // 初始化返回值为空
  AModelSize := 0; // 初始化模型大小为 0

  try
    if ANeedDecrypt then
    begin

    end else begin
      // 打开文件
      FileStream := TFileStream.Create(AFilename, fmOpenRead);
      try
        Result:=FillData(FileStream);
      finally
        FreeAndNil(FileStream);
      end;
    end;
  except
    on E: Exception do
    begin
      // 如果发生异常，输出错误信息并返回 nil
      Result := nil;
      AModelSize := 0;
      raise Exception.Create(Format('Error loading model file %s: %s', [AFilename, E.message]));
    end;
  end;
end;

procedure TRknnBase.SetClassColors;
var
  i,j: Integer;
begin
  FClassColors:=nil;
  SetLength(FClassColors,FClassNames.Count);
  j:=0;
  for i:=0 to FClassNames.Count-1 do
  begin
    FClassColors[i]:=TRectColors[j];
    Inc(j);
    if j > High(TRectColors) then j:=0;
  end;
end;



end.
