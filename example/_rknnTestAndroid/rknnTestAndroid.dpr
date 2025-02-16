program rknnTestAndroid;

uses
  System.StartUpCopy,
  FMX.Forms,
  FMX.Skia,
  rknnTestAndroidMain in 'rknnTestAndroidMain.pas' {rknnTestAndroidMainFrm},
  rknn_api in '..\..\source\rknn_api.pas',
  rknnBase in '..\..\source\rknnBase.pas',
  rknnClassification in '..\..\source\rknnClassification.pas',
  rknnDetection in '..\..\source\rknnDetection.pas';

{$R *.res}

begin
  GlobalUseSkia := True;
  Application.Initialize;
  Application.FormFactor.Orientations := [TFormOrientation.Landscape];
  Application.CreateForm(TrknnTestAndroidMainFrm, rknnTestAndroidMainFrm);
  Application.Run;
end.
