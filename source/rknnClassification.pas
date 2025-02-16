{**************************************************************************

rknn4Delphi (Rockchip RKNN Inference Frame Delphi Wrapped

author: Tom Yea
create time: 2025/1/25
contact: tom_ye@qq.com

本单元提供了一个分类模型的基类和实现的几个分类模型，分类模型不需要做输出张量的解码

****************************************************************************}

unit rknnClassification;

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
  rknnBase,
  rknn_api;

type

  TClassDetResult = record
    ClassId: Word; // 类别ID
    ClassName: string; // 类别名称
    ConfValue: Single; // 置信度
  end;

  TClassDetResults = array of TClassDetResult;

  // =================================================================================
  // 对象分类基类 ========================================================================
  // =================================================================================

  TrknnClassification = class(TrknnBase)
  protected
    FKeepCount: Byte;  //结果保留前多少个
    FResultData: TClassDetResults;

    function DoPostProcess: TArray<Single>; virtual;  //后处理
    procedure FillData(AData: TArray<Single>); virtual;   //填充 FResultData

  public
    constructor Create;
    destructor Destroy; override;

    function LoadModelFile(const AFileName: string; const ANeedDecrypt: Boolean = false): Boolean; override;
    procedure Detect(ABMP: TBitmap); virtual;

  published
    property KeepCount: Byte read FKeepCount write FKeepCount;
    property ResultData: TClassDetResults read FResultData;

  end;

  // ==========================================================================
  // MobileNet分类检测器类
  // ==========================================================================

  TrknnMobileNet = class(TrknnClassification)
  private
    function DoPostProcess: TArray<Single>; //后处理
  public
    procedure Detect(ABMP: TBitmap);
  end;

  // ==========================================================================
  // ResNet分类检测器类
  // ==========================================================================

  TrknnResNet = class (TrknnClassification)
  private
    function DoPostProcess: TArray<Single>; //后处理
  public
    procedure Detect(ABMP: TBitmap);
  end;

implementation

{$R-}

{ TrknnClassification }

constructor TrknnClassification.Create;
begin
  inherited Create;
  FOutputFloat:=true;
  FKeepCount:=5;
end;

destructor TrknnClassification.Destroy;
begin
  inherited Destroy;
end;

procedure TrknnClassification.Detect(ABMP: TBitmap);
begin
  inherited Detect(ABMP);
  FResultData:=nil;
end;

function TrknnClassification.DoPostProcess: TArray<Single>;
var
  i, Index: Integer;
begin
  inherited DoPostprocess;

  // 分类模型只支持1个Tensor输出
  if Length(FOutputs) < 1 then exit;
  if FOutputs[0].size < SizeOf(Single) then exit; // 如果没有数据

  //把输出缓存中的数据读取出来
  SetLength(Result, FOutputTensorsAttr[0].size);// FOutputs[0].size div SizeOf(Single));
  Move(FOutputs[0].buf^,Result[0],FOutputs[0].size);

  // 分类模型只支持1个Tensor输出
  if Length(FOutputs) < 1 then exit;
  if FOutputs[0].size < SizeOf(Single) then exit; // 如果没有数据

  //把输出缓存中的数据读取出来
  SetLength(Result, FOutputTensorsAttr[0].size);// FOutputs[0].size div SizeOf(Single));
  Move(FOutputs[0].buf^,Result[0],FOutputs[0].size);
end;

procedure TrknnClassification.FillData(AData: TArray<Single>);
var
  i, Index: Integer;
begin
  FResultData := nil;
  Index := 0;
  for i := 0 to High(AData) do
  begin
    if AData[i] = 0 then continue;
    SetLength(FResultData, Index + 1);
    FResultData[Index].classId := i;
    FResultData[Index].ClassName := FClassNames[i];
    FResultData[Index].ConfValue := AData[i];
    Inc(Index);
  end;

  // 对结果数据进行降排序
  TArray.Sort<TClassDetResult>(FResultData,
    TComparer<TClassDetResult>.Construct(
    function(const A, B: TClassDetResult): Integer
    begin
      // 降序排序：B.ConfValue 在前，A.ConfValue 在后
      if B.ConfValue > A.ConfValue then
        Result := 1
      else if B.ConfValue < A.ConfValue then
        Result := -1
      else
        Result := 0;
    end));

  //最后保留设定的数量
  SetLength(FResultData,FKeepCount);
end;

function TrknnClassification.LoadModelFile(const AFilename: string; const ANeedDecrypt: Boolean = false): Boolean;
begin
  Result:=inherited LoadModelFile(AFileName,ANeedDecrypt);
  if not Result then exit;
  FClassCount:=FOutputTensorsAttr[0].dims[1];  //获取类别数量
end;

{ TrknnMobileNet }

{ TrknnResNet }

procedure TrknnResNet.Detect(ABMP: TBitmap);
begin
  inherited Detect(ABMP);
  DoPostprocess; // 做后处理
end;

function TrknnResNet.DoPostprocess: TArray<Single>;
begin
  Result:=inherited DoPostprocess;
  Softmax(Result);
  FillData(Result);
end;

{ TrknnMobileNet }

procedure TrknnMobileNet.Detect(ABMP: TBitmap);
begin
  inherited Detect(ABMP);
  DoPostProcess;
end;

function TrknnMobileNet.DoPostProcess: TArray<Single>;
begin
  Result:=inherited DoPostProcess;
  FillData(Result);
end;

end.
