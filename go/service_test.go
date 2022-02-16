package yolov5face_test

import (
	"fmt"
	"time"

	"gitlab.gtvorg.tk/pannixilin/yolov5face/go"
)

func Example() {
	var (
		result  *yolov5face.Result
		service *yolov5face.Service
		imgFile string
		err     error
	)

	imgFile = "test/zidane.jpg"

	// service only need to be created once
	service = yolov5face.NewHttpService(
		"http://localhost:8080/predictions/fd1",
		time.Millisecond * time.Duration(5000),
		1920,
		1080,
	)
	if !service.IsAlive() {
		fmt.Println("service is not availabe")
		return
	}

	if result, err = service.DetectHeadFromFile(imgFile); err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("image file: %s\n", imgFile)
	fmt.Printf("image width: %d\n", result.Width)
	fmt.Printf("image height: %d\n", result.Height)
	for i, h := range result.Heads {
		fmt.Printf(
			"head[%d]: Box[x:%d, y:%d, x:%d, y:%d], Score: %f\n",
			i, h.Box[0], h.Box[1], h.Box[2], h.Box[3], h.Score)
	}

	// Output:
	// image file: test/zidane.jpg
	// image width: 1280
	// image height: 720
	// head[0]: Box[x:910, y:91, x:1058, y:279], Score: 0.864714
	// head[1]: Box[x:518, y:236, x:672, y:448], Score: 0.726488
}
