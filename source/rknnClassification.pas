{**************************************************************************

rknn4Delphi (Rockchip RKNN Inference Frame Delphi Wrapped

author: Tom Yea
create time: 2025/1/25
contact: tom_ye@qq.com

����Ԫ�ṩ��һ������ģ�͵Ļ����ʵ�ֵļ�������ģ�ͣ�����ģ�Ͳ���Ҫ����������Ľ���

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
    ClassId: Word; // ���ID
    ClassName: string; // �������
    ConfValue: Single; // ���Ŷ�
  end;

  TClassDetResults = array of TClassDetResult;

  // =================================================================================
  // ���������� ========================================================================
  // =================================================================================

  TrknnClassification = class(TrknnBase)
  protected
    FKeepCount: Byte;  //�������ǰ���ٸ�
    FResultData: TClassDetResults;

    function DoPostProcess: TArray<Single>; virtual;  //����
    procedure FillData(AData: TArray<Single>); virtual;   //��� FResultData

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
  // MobileNet����������
  // ==========================================================================

  TrknnMobileNet = class(TrknnClassification)
  private
    function DoPostProcess: TArray<Single>; //����
  public
    procedure Detect(ABMP: TBitmap);
  end;

  // ==========================================================================
  // ResNet����������
  // ==========================================================================

  TrknnResNet = class (TrknnClassification)
  private
    function DoPostProcess: TArray<Single>; //����
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

  // ����ģ��ֻ֧��1��Tensor���
  if Length(FOutputs) < 1 then exit;
  if FOutputs[0].size < SizeOf(Single) then exit; // ���û������

  //����������е����ݶ�ȡ����
  SetLength(Result, FOutputTensorsAttr[0].size);// FOutputs[0].size div SizeOf(Single));
  Move(FOutputs[0].buf^,Result[0],FOutputs[0].size);

  // ����ģ��ֻ֧��1��Tensor���
  if Length(FOutputs) < 1 then exit;
  if FOutputs[0].size < SizeOf(Single) then exit; // ���û������

  //����������е����ݶ�ȡ����
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

  // �Խ�����ݽ��н�����
  TArray.Sort<TClassDetResult>(FResultData,
    TComparer<TClassDetResult>.Construct(
    function(const A, B: TClassDetResult): Integer
    begin
      // ��������B.ConfValue ��ǰ��A.ConfValue �ں�
      if B.ConfValue > A.ConfValue then
        Result := 1
      else if B.ConfValue < A.ConfValue then
        Result := -1
      else
        Result := 0;
    end));

  //������趨������
  SetLength(FResultData,FKeepCount);
end;

function TrknnClassification.LoadModelFile(const AFilename: string; const ANeedDecrypt: Boolean = false): Boolean;
begin
  Result:=inherited LoadModelFile(AFileName,ANeedDecrypt);
  if not Result then exit;
  FClassCount:=FOutputTensorsAttr[0].dims[1];  //��ȡ�������
end;

{ TrknnMobileNet }

{ TrknnResNet }

procedure TrknnResNet.Detect(ABMP: TBitmap);
begin
  inherited Detect(ABMP);
  DoPostprocess; // ������
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
