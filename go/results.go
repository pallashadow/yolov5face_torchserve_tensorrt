package yolov5face


type Result struct {
	Width  int
	Height int
	Heads  []Head
}

type Head struct {
	Box   [4]int
	Score float32
}

type HttpResult struct {
	Xywh_ratio      []float32                `json:"xywh_ratio"`
	Conf            float32                  `json:"conf"`
	Landmarks_ratio []float32                `json:"landmarks_ratio"`
}
