package yolov5face

import (
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
)

func HttpResult2Result(width int, height int, httpResults []HttpResult) *Result {
	var res = Result{Width: width, Height: height}
	for _, v := range httpResults {
		centr_x := int(float32(width) * v.Xywh_ratio[0])
		centr_y := int(float32(height) * v.Xywh_ratio[1])
		head_width := int(float32(width) * v.Xywh_ratio[2])
		head_height := int(float32(height) * v.Xywh_ratio[3])
		upLeft_x := centr_x - head_width / 2
		upLeft_y := centr_y - head_height / 2
		bottomRight_x := centr_x + head_width / 2
		bottomRight_y := centr_y + head_height / 2
		h := Head{
			Box: [4]int{upLeft_x, upLeft_y, bottomRight_x, bottomRight_y},
			Score: v.Conf,
		}
		res.Heads = append(res.Heads, h)
	}
	return &res
}

func GetImageSize(img io.Reader) (width int, height int, err error) {
	if image, _, err := image.DecodeConfig(img); err == nil {
		width = image.Width
		height = image.Height
	} else {
		width = 0
		height = 0
	}
	return
}

// unify the width and height to the size of a horizontal image
func unifyImageWidthHeight(width int, height int) (width_unified int, height_unified int) {
	if width >= height {
		width_unified = width
		height_unified = height
	} else {
		width_unified = height
		height_unified = width
	}
	return
}
