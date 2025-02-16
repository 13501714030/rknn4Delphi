unit rknnTestAndroidMain;

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes, System.Variants,
  FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics, FMX.Dialogs, FMX.Controls.Presentation, FMX.StdCtrls,
  FMX.Memo.Types, FMX.Layouts,
  FMX.ScrollBox, FMX.Memo, System.Threading, System.IOUtils,
  FMX.Objects, System.Diagnostics,
  System.Permissions,
  System.StrUtils,
  Androidapi.JNI.JavaTypes,
  Androidapi.JNI.Os,
  Androidapi.Helpers,
  Androidapi.JNIBridge,
  Androidapi.JNI.GraphicsContentViewText,
  Androidapi.JNI.App,
  FMX.Helpers.Android, FMX.TabControl,
  FMX.ListBox,

  rknnBase,
  rknnClassification,
  rknnDetection;

type
  TrknnTestAndroidMainFrm = class(TForm)
    Layout2: TLayout;
    TabControl1: TTabControl;
    TabItem1: TTabItem;
    TabItem2: TTabItem;
    Image1: TImage;
    Image2: TImage;
    Layout3: TLayout;
    Button1: TButton;
    Layout4: TLayout;
    Button3: TButton;
    Layout5: TLayout;
    Memo1: TMemo;
    Layout1: TLayout;
    Button2: TButton;
    Button4: TButton;
    ComboBox2: TComboBox;
    Button8: TButton;
    Splitter1: TSplitter;
    Button9: TButton;
    ComboBox1: TComboBox;
    procedure FormDestroy(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Button3Click(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure Button4Click(Sender: TObject);
    procedure ComboBox1Change(Sender: TObject);
    procedure Button8Click(Sender: TObject);
    procedure Button9Click(Sender: TObject);
  private

    {$define RK3588}
//    {$define RK3576}
//    {$define RK3566}

    { Private declarations }
    {$ifdef RK3588}
      FrknnClassMobileNet: TrknnMobileNet;
    {$endif}
    FrknnYoloV8_V11: TrknnYoloV8Detection;

  public
    { Public declarations }
  end;

var
  rknnTestAndroidMainFrm: TrknnTestAndroidMainFrm;

implementation

{$R *.fmx}

uses MiscFmx;

procedure TrknnTestAndroidMainFrm.Button1Click(Sender: TObject);
var
  i: Integer;
  Stopwatch: TStopwatch;
  ElapsedMicroseconds: Int64;
  s: string;
begin
  TabControl1.ActiveTab:=TabItem1;
  Memo1.Lines.Clear;
  Stopwatch := TStopwatch.StartNew;
  FrknnYoloV8_V11.Detect(Image2.Bitmap);
  ElapsedMicroseconds := Stopwatch.ElapsedTicks * 1000000 div TStopwatch.Frequency;
  s:=ExtractFileName(FrknnYoloV8_V11.ModelFile);
  s:=s+' = count: '+FormatFloat('00',FrknnYoloV8_V11.ResultCount);
  s:=s+';  '+FormatFloat('0.00 ms',ElapsedMicroseconds / 1000);
  s:=s+FormatFloat(' ( 0.0 fps )',1000 / (ElapsedMicroseconds / 1000));
  FrknnYoloV8_V11.DrawResult(Image2.Bitmap,s);
end;

procedure TrknnTestAndroidMainFrm.Button8Click(Sender: TObject);
begin
{$ifdef RK3588}
  Memo1.Lines.Clear;
  Memo1.Lines.Add('model file = '+ExtractFileName(FrknnClassMobileNet.ModelFile));
  Memo1.Lines.Add('quantized = '+Integer(FrknnClassMobileNet.Quantized).ToString+'; class count = '+FrknnClassMobileNet.ClassCount.ToString);
  Memo1.Lines.Add('input image size = '+FrknnClassMobileNet.InputImgWidth.ToString+' x '+FrknnClassMobileNet.InputImgHeight.ToString);
  Memo1.Lines.Add('');
  Memo1.Lines.Add(FrknnClassMobileNet.GetTensorsInfo);
{$endif}
end;

procedure TrknnTestAndroidMainFrm.Button9Click(Sender: TObject);
begin
  Image2.Bitmap.LoadFromFile(System.IOUtils.TPath.Combine(System.IOUtils.TPath.GetHomePath,'p720_01.bmp'));
end;

procedure TrknnTestAndroidMainFrm.ComboBox1Change(Sender: TObject);
var
  s: string;
begin
  if not Assigned(FrknnYoloV8_V11) then exit;  
  s:=System.IOUtils.TPath.Combine(System.IOUtils.TPath.GetHomePath,ComboBox1.Text);
  if not FileExists(s) then
  begin
    ShowMessage(s+' does not exist !');
    exit;
  end;
  try
    FrknnYoloV8_V11.LoadModelFile(s);
  except
    on E: Exception do ShowMessage(E.Message);
  end;
end;

procedure TrknnTestAndroidMainFrm.Button2Click(Sender: TObject);
begin
  ShowMessage(Memo1.Text);
end;

procedure TrknnTestAndroidMainFrm.Button3Click(Sender: TObject);
var
  i: Integer;
begin
{$ifdef RK3588}
  Memo1.Lines.Clear;
  FrknnClassMobileNet.Detect(Image1.Bitmap);
  for i:=0 to High(FrknnClassMobileNet.ResultData) do
    Memo1.Lines.Add(FrknnClassMobileNet.ResultData[i].ConfValue.ToString+' = '+
      FrknnClassMobileNet.ResultData[i].ClassID.ToString + ' '+FrknnClassMobileNet.ResultData[i].ClassName);
{$endif}
end;

procedure TrknnTestAndroidMainFrm.Button4Click(Sender: TObject);
begin
  Memo1.Lines.Clear;
  Memo1.Lines.Add('model file = '+ExtractFileName(FrknnYoloV8_V11.ModelFile));
  Memo1.Lines.Add('quantized = '+Integer(FrknnYoloV8_V11.Quantized).ToString+'; class count = '+FrknnYoloV8_V11.ClassCount.ToString);
  Memo1.Lines.Add('input image size = '+FrknnYoloV8_V11.InputImgWidth.ToString+' x '+FrknnYoloV8_V11.InputImgHeight.ToString);
  Memo1.Lines.Add('');
  Memo1.Lines.Add(FrknnYoloV8_V11.GetTensorsInfo);
end;

procedure TrknnTestAndroidMainFrm.FormCreate(Sender: TObject);
var
  i: Integer;
begin
  try

    {$ifdef RK3588}
    FrknnClassMobileNet:=TrknnMobileNet.Create;
//    FrknnClassMobileNet.LogMemo:=Memo1;
    FrknnClassMobileNet.LoadModelFile(System.IOUtils.TPath.Combine(System.IOUtils.TPath.GetHomePath,'mobilenet_v1.rknn'));
    FrknnClassMobileNet.LoadClassNames(System.IOUtils.TPath.Combine(System.IOUtils.TPath.GetHomePath,'mobilenet_v1_classes.txt'));

    {$endif}

    {$ifdef RK3566}
      for i:=ComboBox1.Items.Count -1 downto 0 do
        if not ComboBox1.Items[i].Contains('_3566') then ComboBox1.Items.Delete(i);
    {$endif}
    {$ifdef RK3576}
      for i:=ComboBox1.Items.Count -1 downto 0 do
        if not ComboBox1.Items[i].Contains('_3576') then ComboBox1.Items.Delete(i);
    {$endif}
    {$ifdef RK3588}
      for i:=ComboBox1.Items.Count -1 downto 0 do
        if not ComboBox1.Items[i].Contains('_3588') then ComboBox1.Items.Delete(i);
    {$endif}
    ComboBox1.ItemIndex:=0;
    FrknnYoloV8_V11:=TrknnYoloV8Detection.Create;
    FrknnYoloV8_V11.LoadModelFile(System.IOUtils.TPath.Combine(System.IOUtils.TPath.GetHomePath,ComboBox1.Text));
    FrknnYoloV8_V11.LoadClassNames(System.IOUtils.TPath.Combine(System.IOUtils.TPath.GetHomePath,'vd_classes.txt'));
  except
    on E: Exception do
      ShowMessage(E.Message);
  end;

end;

procedure TrknnTestAndroidMainFrm.FormDestroy(Sender: TObject);
begin
{$ifdef RK3588}
  FreeAndNil(FrknnClassMobileNet);
{$endif}
  FreeAndNil(FrknnYoloV8_V11);
end;

end.
