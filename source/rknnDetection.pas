{**************************************************************************

rknn4Delphi (Rockchip RKNN Inference Frame Delphi Wrapped

author: Tom Yea
create time: 2025/1/25
contact: tom_ye@qq.com

本单元为rknn对象检测单元，提供了所有对象检测的类
先定义了一个对象检测的基类：TrknnDetection ，继承自 rknnBase 基类
此类提供了一些最基础的对象检测任务需要用到的参数等设置
由于对象识别模型有很多，较为流行的为Yolo，具体信息可查看
https://github.com/airockchip/rknn_model_zoo

目前作者只实现了ultralytics的 YoloV8 (同时也兼容 YoloV11)
https://github.com/ultralytics/ultralytics
其余的对象识别检测可以从TrknnDetection继承后自行增加

****************************************************************************}

unit rknnDetection;

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

  // 对象识别 =======================================================================
  TObjDetResult = record
    TrackID: Integer;                                 //Deepsort后的ID
    ClassId: Word; // 类别ID
    ClassName: string; // 类别名称
    ConfValue: Single; // 置信度
    Rect: TRect;                                      //框坐标位置
    function Left: Integer;
    function Top: Integer;
    function Right: Integer;
    function Bottom: Integer;
    function Width: Integer;
    function Height: Integer;
    function cX: Integer;
    function cY: Integer;
    function Area: Integer;
    //获取笛卡尔坐标数据
    procedure GetCartesianCoordinates(const AWidth,AHeight: Integer; var x,y: SmallInt);
  end;

  TObjDetResults = array of TObjDetResult;

  TrknnDetection = class(TRknnBase)
  private

  protected
    FKeepOnlyMaxConf: Boolean;      //是否只要最大置信度的，此属性设为true后，ResultData只会有一个数据
    FResultData: TObjDetResults;

  public
    constructor Create;
    destructor Destroy; override;

    function LoadModelFile(const AFilename: string; const ANeedDecrypt: Boolean = false): Boolean; override;
    procedure DrawResult(ABitmap: TBitmap; const AInfo: string; const ADrawLabel: Boolean = true); //画框
    function ResultCount: Integer;   //识别出来的对象总数量

  published
    property KeepOnlyMaxConf: Boolean read FKeepOnlyMaxConf write FKeepOnlyMaxConf;
    property ResultData: TObjDetResults read FResultData;

  end;

  // ==========================================================================
  //YoloV8检测器类
  TrknnYoloV8Detection = class (TrknnDetection)
  protected

    // yolov8解码坐标框，置信度和类别等
    procedure YoloV8Decode;

    function process_i8(box_tensor: TArray<Int8>; box_zp: Integer; box_scale: Single;
                        score_tensor: TArray<Int8>; score_zp: Integer; score_scale: Single;
                        score_sum_tensor: TArray<Int8>; score_sum_zp: Integer; score_sum_scale: Single;
                        grid_h, grid_w, stride, dfl_len: Integer;
                        var boxes: TArray<Single>; var objProbs: TArray<Single>; var classId: TArray<Integer>;
                        threshold: Single): Integer;

    function process_f32(box_tensor, score_tensor, score_sum_tensor: TArray<Single>;
                         grid_h, grid_w, stride, dfl_len: Integer;
                         var boxes: TArray<Single>; var objProbs: TArray<Single>; var classId: TArray<Integer>;
                         threshold: Single): Integer;

    procedure DoPostProcess; virtual;

  public
    procedure Detect(ABMP: TBitmap); // 识别
  end;

  // ==========================================================================
  // YoloV11检测器类
  //目前还没发现不兼容YoloV8的地方，可以直接兼容
  //后续如果由修改，可以在这里对YoloV11做改动
  TrknnYoloV11Detection = class (TrknnYoloV8Detection)

  end;

implementation

{$R-}

{ TObjDetResult }

function TObjDetResult.Area: Integer;
begin
  Result:=Rect.Width * Rect.Height;
end;

function TObjDetResult.Bottom: Integer;
begin
  Result:=Rect.Bottom;
end;

function TObjDetResult.cX: Integer;
begin
  Result:=Rect.CenterPoint.X;
end;

function TObjDetResult.cY: Integer;
begin
  Result:=Rect.CenterPoint.Y;
end;

procedure TObjDetResult.GetCartesianCoordinates(const AWidth, AHeight: Integer; var x, y: SmallInt);
begin
  x:=Self.cX - AWidth div 2;
  y:=-(Self.cY - AHeight div 2);
end;

function TObjDetResult.Height: Integer;
begin
  Result:=Rect.Height;
end;

function TObjDetResult.Left: Integer;
begin
  Result:=Rect.Left;
end;

function TObjDetResult.Right: Integer;
begin
  Result:=Rect.Right;
end;

function TObjDetResult.Top: Integer;
begin
  Result:=Rect.Top;
end;

function TObjDetResult.Width: Integer;
begin
  Result:=Rect.Width;
end;


{ TRknn_ObjDet }

constructor TrknnDetection.Create;
begin
  inherited Create;
  FOutputFloat:=false;
  FKeepOnlyMaxConf:=false;
end;

destructor TrknnDetection.Destroy;
begin

  inherited Destroy;
end;

function TrknnDetection.ResultCount: Integer;
begin
  Result:=Length(FResultData);
end;

procedure TrknnDetection.DrawResult(ABitmap: TBitmap; const AInfo: string; const ADrawLabel: Boolean = true);
var
  i,AWidth,AHeight,AFontSize: Integer;
  AThickness: Single;
  ARect: TRectF;
  ALabel: string;
  AFrameColor,AFontColor: TAlphaColor;
  ACanvas: TCanvas;
  rData: TObjDetResults;
begin
  inherited DrawResult(ABitmap,AInfo,ADrawLabel);
  ACanvas:=ABitmap.Canvas;

  AThickness:=ABitmap.Width / 400;
  AFontSize:=Round(ABitmap.Width / 64);

  if not ACanvas.BeginScene() then exit;

  ACanvas.Stroke.Kind:=TBrushKind.Solid;
  ACanvas.Stroke.Thickness := AThickness;
  ACanvas.Font.Size:=AFontSize;
  try
    rData:=ResultData;

    for i:=0 to High(rData) do
    begin
      ARect.Left:=rData[i].Rect.Left;
      ARect.Top:=rData[i].Rect.Top;
      ARect.Width:=rData[i].Rect.Width;
      ARect.Height:=rData[i].Rect.Height;
      ALabel:=Copy(rData[i].ClassName,rData[i].ClassName.IndexOf('=')+2,
        Length(rData[i].ClassName))+FormatFloat(' 0.0%',rData[i].ConfValue * 100);//+' '+ResultData[i].ClassID.ToString;

      if rData[i].ClassID < Length(FClassColors) then
        AFrameColor:=FClassColors[rData[i].ClassID]
      else
        AFrameColor:=TAlphaColorRec.Lime;

      ACanvas.Stroke.Color:=AFrameColor;
      ACanvas.Fill.Color:=AFrameColor;
      ACanvas.DrawRect(ARect, 0, 0, AllCorners, 1);
      if FDrawType = dtFilled then
        ACanvas.FillRect(ARect, 0, 0, AllCorners, 0.15);

      if ADrawLabel then //是否要画label
      begin
        if (AFrameColor=TAlphaColorRec.Seagreen)  or (AFrameColor=TAlphaColorRec.Sandybrown) or
           (AFrameColor=TAlphaColorRec.Royalblue) or (AFrameColor=TAlphaColorRec.Blueviolet) or (AFrameColor=TAlphaColorRec.Tomato) or
           (AFrameColor=TAlphaColorRec.Saddlebrown) or (AFrameColor=TAlphaColorRec.Cornflowerblue) then
          AFontColor:=TAlphaColorRec.White
        else
          AFontColor:=TAlphaColorRec.Black;

        AHeight:=Round(ACanvas.TextHeight(ALabel) * 1.1);
        AWidth:=Round(ACanvas.TextWidth(ALabel) * 1.05);

        if ARect.Top < AHeight then
          ARect.Top:=ARect.Bottom;

        ARect.Bottom:=ARect.Top;
        ARect.Top:=ARect.Top - AHeight;
        ARect.Right:=ARect.Left + AWidth;
        ARect.Left:=Round(ARect.Left * 0.995);

        ACanvas.Fill.Color:=AFrameColor;
        ACanvas.FillRect(ARect, 0 ,0, AllCorners, 1);

        ACanvas.Fill.Color:=AFontColor;
        ACanvas.FillText(ARect, ALabel, False, 1, [], TTextAlign.Center, TTextAlign.Center );
      end;
    end;

    if AInfo = '' then exit;      //是否要画帧率等信息

    AFontSize:=AFontSize + 10;
    ACanvas.Font.Size:=AFontSize;
    ACanvas.Font.Style:=ACanvas.Font.Style+[TFontStyle.fsBold];
    ARect.Left:=5;
    ARect.Top:=3;

    AHeight:=Round(ACanvas.TextHeight(AInfo)) + 4;
    AWidth:=Round(ACanvas.TextWidth(AInfo)) + 4;

    ARect.Right:=ARect.Left + AWidth;
    ARect.Bottom:=ARect.Top + AHeight;

    ACanvas.Fill.Color:=TAlphaColorRec.Black;
    ACanvas.FillText(ARect, AInfo, False, 1, [], TTextAlign.Leading, TTextAlign.Center);

    ARect.Left:=ARect.Left + 3;
    ARect.Top:=ARect.Top + 3;
    ACanvas.FillText(ARect, AInfo, False, 1, [], TTextAlign.Leading, TTextAlign.Center);

    ARect.Left:=ARect.Left - 2;
    ARect.Top:=ARect.Top - 2;
    ACanvas.Fill.Color:=TAlphaColorRec.White;// TAlphaColorRec.Springgreen;
    ACanvas.FillText(ARect, AInfo, False, 1, [], TTextAlign.Leading, TTextAlign.Center);

  finally
    ACanvas.EndScene;
  end;
end;

function TrknnDetection.LoadModelFile(const AFilename: string; const ANeedDecrypt: Boolean = false): Boolean;
begin
  Result:=inherited LoadModelFile(AFileName,ANeedDecrypt);
  if not Result then exit;
  FClassCount:=FOutputTensorsAttr[1].dims[1];  //获取类别数量
end;

{TrknnYoloV8Detection}

procedure TrknnYoloV8Detection.Detect(ABMP: TBitmap);
begin
  if FIOCount.n_output <> 9 then
    raise Exception.Create('It is not a valid RKNN yoloV8 model !');
  FResultData:=nil;
  inherited Detect(ABMP);
  DoPostProcess;
end;

procedure TrknnYoloV8Detection.DoPostProcess;
begin
  inherited DoPostProcess;
  YoloV8Decode;
end;

procedure TrknnYoloV8Detection.YoloV8Decode;
var
  filterBoxes, objProbs: TArray<Single>;
  classId, TmpClassId: TArray<Integer>;
  validCount, stride, grid_h, grid_w, model_in_w, model_in_h: Integer;
  dfl_len, output_per_branch, box_idx, score_idx, score_sum_idx: Integer;
  box_zp,score_zp,score_sum_zp: Integer;
  box_scale,score_scale,score_sum_scale: Single;
  indexArray: TArray<Integer>;
  logi,i, n, last_count, id, Index: Integer;
  x1, y1, x2, y2, obj_conf: Single;
  box_tensor, score_tensor, score_sum_tensor: TArray<Int8>;
  box_tensor_f, score_tensor_f, score_sum_tensor_f: TArray<Single>;
  s: string;
begin
  model_in_w := Self.InputImgWidth;
  model_in_h := Self.InputImgHeight;

  dfl_len := FOutputTensorsAttr[0].dims[1] div 4;
  output_per_branch := Self.FIOCount.n_output div 3;
  validCount:=0;
  for i := 0 to 2 do
  begin
    box_zp:=0; score_zp:=0; score_sum_zp := 0;
    box_scale:=1.0; score_scale:=1.0; score_sum_scale := 1.0;

    //获取tensor索引号
    box_idx := i * output_per_branch;
    score_idx := i * output_per_branch + 1;
    score_sum_idx := i * output_per_branch + 2;

    //获取tensor数据
    box_tensor := nil; box_tensor_f := nil;
    box_zp:=FOutputTensorsAttr[box_idx].zp;
    box_scale:=FOutputTensorsAttr[box_idx].scale;
    if FIsQuantized then  //如果是量化的模型
    begin
      SetLength(box_tensor, FOutputTensorsAttr[box_idx].size);
      Move(FOutputs[box_idx].buf^, box_tensor[0], FOutputs[box_idx].size);// FOutputTensorsAttr[box_idx].size * SizeOf(Int8));
    end else begin
      SetLength(box_tensor_f, FOutputTensorsAttr[box_idx].size);
      Move(FOutputs[box_idx].buf^, box_tensor_f[0], FOutputs[box_idx].size);// FOutputTensorsAttr[box_idx].size * SizeOf(Int8));
    end;
    score_tensor := nil;
    score_zp:=FOutputTensorsAttr[score_idx].zp;
    score_scale:=FOutputTensorsAttr[score_idx].scale;
    if FIsQuantized then  //如果是量化的模型
    begin
      SetLength(score_tensor, FOutputTensorsAttr[score_idx].size);
      Move(FOutputs[score_idx].buf^, score_tensor[0], FOutputs[score_idx].size); // FOutputTensorsAttr[score_idx].size * SizeOf(Int8));
    end else begin
      SetLength(score_tensor_f, FOutputTensorsAttr[score_idx].size);
      Move(FOutputs[score_idx].buf^, score_tensor_f[0], FOutputs[score_idx].size); // FOutputTensorsAttr[score_idx].size * SizeOf(Int8));
    end;
    score_sum_tensor := nil;
    score_sum_zp := FOutputTensorsAttr[score_sum_idx].zp;
    score_sum_scale := FOutputTensorsAttr[score_sum_idx].scale;
    if FIsQuantized then  //如果是量化的模型
    begin
      SetLength(score_sum_tensor, FOutputTensorsAttr[score_sum_idx].size);
      Move(FOutputs[score_sum_idx].buf^, score_sum_tensor[0], FOutputs[score_sum_idx].size); // * SizeOf(Int8));
    end else begin
      SetLength(score_sum_tensor_f, FOutputTensorsAttr[score_sum_idx].size);
      Move(FOutputs[score_sum_idx].buf^, score_sum_tensor_f[0], FOutputs[score_sum_idx].size); // * SizeOf(Int8));
    end;
    //获取grid和stride
    grid_h := FOutputTensorsAttr[box_idx].dims[2];
    grid_w := FOutputTensorsAttr[box_idx].dims[3];
    stride := model_in_h div grid_h;
    //进入循环i8计算
    if FIsQuantized then
      validCount := validCount + process_i8(box_tensor, box_zp, box_scale,
                                            score_tensor, score_zp, score_scale,
                                            score_sum_tensor, score_sum_zp, score_sum_scale,
                                            grid_h, grid_w, stride, dfl_len,
                                            filterBoxes, objProbs, classId, FConfThres)
    else begin
      validCount:=validCount + process_f32(box_tensor_f,score_tensor_f, score_sum_tensor_f,
                                           grid_h, grid_w, stride, dfl_len,
                                           filterBoxes, objProbs, classId, FConfThres);
    end;

  end;
  if validCount <= 0 then exit;
  SetLength(indexArray, validCount);
  for i := 0 to validCount - 1 do indexArray[i] := i;
   //排序
  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
  //去重
  TmpClassId:=TArrayHelper.CreateRemoveDuplicates<Integer>(classId);
  //去重后再排序
  TArrayHelper.SortArray<Integer>(TmpClassId,0,true);
  //做nums
  for i:=0 to High(TmpClassId) do nms(validCount, filterBoxes, classId, indexArray, TmpClassId[i], FNMSThres);
  last_count := 0;
  //初始化 od_results
  FResultData:=nil; Index:=0;
  if FKeepOnlyMaxConf then SetLength(FResultData,1);  //如果只要最大置信度值的只要设置1位即可

  // box valid detect target
  for i := 0 to validCount - 1 do
  begin
    if (indexArray[i] = -1) or (last_count >= OBJ_NUMB_MAX_SIZE) then Continue;

    n := indexArray[i];

    x1 := filterBoxes[n * 4 + 0];// - ALetterBox.x_pad;
    y1 := filterBoxes[n * 4 + 1];// - ALetterBox.y_pad;
    x2 := x1 + filterBoxes[n * 4 + 2];
    y2 := y1 + filterBoxes[n * 4 + 3];
    id := classId[n];
    obj_conf := objProbs[i];

    if obj_conf > 0 then
    begin
      if FExcludeClassIDs.IndexOf(id.ToString) < 0 then
      begin
        if FKeepOnlyMaxConf then
        begin
          if obj_conf > FResultData[0].ConfValue then
          begin
            FResultData[0].TrackID:=-1;
            FResultData[0].ClassId:=id;
            if (id < FClassNames.Count) then
              FResultData[0].ClassName:=FClassNames[id]
            else
              FResultData[0].ClassName:='NA';
            FResultData[0].ConfValue:=obj_conf;
            FResultData[0].Rect.Left:=Trunc(clamp(x1, 0, model_in_w) * Self.FImgW_rate);
            FResultData[0].Rect.Top:=Trunc(clamp(y1, 0, model_in_h) * Self.FImgH_rate);
            FResultData[0].Rect.Right:=Trunc(clamp(x2, 0, model_in_w) * Self.FImgW_rate);
            FResultData[0].Rect.Bottom:=Trunc(clamp(y2, 0, model_in_h) * Self.FImgH_rate);
          end;
        end else begin
          SetLength(FResultData,Index+1);
          FResultData[Index].TrackID:=-1;
          FResultData[Index].ClassId:=id;
          if (id < FClassNames.Count) then
            FResultData[Index].ClassName:=FClassNames[id]
          else
            FResultData[Index].ClassName:='NA';
          FResultData[Index].ConfValue:=obj_conf;
          FResultData[Index].Rect.Left:=Trunc(clamp(x1, 0, model_in_w) * Self.FImgW_rate);
          FResultData[Index].Rect.Top:=Trunc(clamp(y1, 0, model_in_h) * Self.FImgH_rate);
          FResultData[Index].Rect.Right:=Trunc(clamp(x2, 0, model_in_w) * Self.FImgW_rate);
          FResultData[Index].Rect.Bottom:=Trunc(clamp(y2, 0, model_in_h) * Self.FImgH_rate);
          Inc(Index);
        end;
      end;
    end;
    Inc(last_count);
  end;
end;

function TrknnYoloV8Detection.process_f32(box_tensor, score_tensor, score_sum_tensor: TArray<Single>;
                                          grid_h, grid_w, stride, dfl_len: Integer;
                                          var boxes, objProbs: TArray<Single>; var classId: TArray<Integer>;
                                          threshold: Single): Integer;
var
  validCount: Integer;
  offset, max_class_id, grid_len: Integer;
  max_score: Single;
  box: TboxArray;
  before_dfl: TArray<Single>;
  x1,y1,x2,y2,w,h: Single;
  box_idx,obj_idx,class_idx,k,c,i,j: Integer;
begin
  validCount:=0;
  grid_len := grid_h * grid_w;

  box_idx:=Length(boxes);
  obj_idx:=Length(objProbs);
  class_idx:=Length(classId);

  for i:=0 to grid_h - 1 do
  begin
    for j:=0 to grid_w - 1 do
    begin
      offset := i * grid_w + j;
      max_class_id := -1;
      // 通过 score sum 起到快速过滤的作用
      if (score_sum_tensor <> nil) then
        if (score_sum_tensor[offset] < threshold) then continue;
      max_score:=0;
      for c := 0 to FClassCount - 1 do
      begin
        if (score_tensor[offset] > threshold) and (score_tensor[offset] > max_score) then
        begin
          max_score := score_tensor[offset];
          max_class_id := c;
        end;
        offset := offset + grid_len;
      end;
      //compute box
      if (max_score> threshold) then
      begin
        offset := i* grid_w + j;
        SetLength(before_dfl, dfl_len * 4);
        for k:=0 to dfl_len * 4 - 1 do
        begin
          before_dfl[k] := box_tensor[offset];
          offset := offset + grid_len;
        end;
        compute_dfl(before_dfl, dfl_len, box);

        x1 := (-box[0] + j + 0.5)*stride;
        y1 := (-box[1] + i + 0.5)*stride;
        x2 := (box[2] + j + 0.5)*stride;
        y2 := (box[3] + i + 0.5)*stride;
        w := x2 - x1;
        h := y2 - y1;

        SetLength(boxes, box_idx + 4);
        SetLength(objProbs, obj_idx + 1);
        SetLength(classId, class_idx + 1);

        boxes[box_idx] := x1;
        boxes[box_idx+1] := y1;
        boxes[box_idx+2] := w;
        boxes[box_idx+3] := h;
        objProbs[obj_idx] := max_score;
        classId[class_idx] := max_class_id;
        Inc(validCount);
      end;
    end;
  end;
  Result:=validCount;
end;

function TrknnYoloV8Detection.process_i8(box_tensor: TArray<Int8>; box_zp: Integer; box_scale: Single;
                                 score_tensor: TArray<Int8>; score_zp: Integer; score_scale: Single;
                                 score_sum_tensor: TArray<Int8>; score_sum_zp: Integer; score_sum_scale: Single;
                                 grid_h, grid_w, stride, dfl_len: Integer;
                                 var boxes, objProbs: TArray<Single>; var classId: TArray<Integer>;
                                 threshold: Single): Integer;
var
  logi,box_idx,obj_idx,class_idx, c, k, i, j, validCount: Integer;
  offset, max_class_id, grid_len: Integer;
  max_score: Int8;
  score_thres_i8: Int8;
  score_sum_thres_i8: Int8;
  x1, y1, x2, y2, w, h: Single;
  box: TboxArray;
  before_dfl: TArray<Single>;
begin
  validCount := 0;
  grid_len := grid_h * grid_w;
  score_thres_i8 := qnt_f32_to_affine(threshold, score_zp, score_scale);
  score_sum_thres_i8 := qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);
  before_dfl:=nil;
  SetLength(before_dfl,dfl_len * 4);

  box_idx:=Length(boxes);
  obj_idx:=Length(objProbs);
  class_idx:=Length(classId);

  for i := 0 to grid_h - 1 do
  begin
    for j := 0 to grid_w - 1 do
    begin
      offset := i * grid_w + j;
      max_class_id := -1;

      // 通过 score sum 起到快速过滤的作用
      if (score_sum_tensor <> nil) then
        if (score_sum_tensor[offset] < score_sum_thres_i8) then continue;

      //compute class confs
      max_score := -score_zp;
      for c := 0 to FClassCount - 1 do
      begin
        if (score_tensor[offset] > score_thres_i8) and
           (score_tensor[offset] > max_score) then
        begin
          max_score := score_tensor[offset];
          max_class_id := c;
        end;
        offset := offset + grid_len;
      end;

      // compute box
      if (max_score > score_thres_i8) then
      begin
        offset := i * grid_w + j;
        for k := 0 to dfl_len * 4 - 1 do
        begin
          before_dfl[k] := deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
          offset := offset + grid_len;
        end;
        compute_dfl(before_dfl, dfl_len, box);

        x1 := (-box[0] + j + 0.5) * stride;
        y1 := (-box[1] + i + 0.5) * stride;
        x2 := (box[2] + j + 0.5) * stride;
        y2 := (box[3] + i + 0.5) * stride;
        w := x2 - x1;
        h := y2 - y1;

        SetLength(boxes, box_idx + 4);
        SetLength(objProbs, obj_idx + 1);
        SetLength(classId, class_idx + 1);

        boxes[box_idx] := x1;
        boxes[box_idx+1] := y1;
        boxes[box_idx+2] := w;
        boxes[box_idx+3] := h;
        objProbs[obj_idx] := deqnt_affine_to_f32(max_score, score_zp, score_scale);
        classId[class_idx] := max_class_id;

        Inc(validCount);
        Inc(box_idx,4);
        Inc(obj_idx);
        Inc(class_idx);
      end;
    end;
  end;
  Result := validCount;
end;


end.
