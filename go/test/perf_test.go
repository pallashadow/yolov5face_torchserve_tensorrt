package unitests

import (
	"io/ioutil"
	"os"
	"testing"
	"time"

	"gitlab.gtvorg.tk/pannixilin/yolov5face/go"
)

func BenchmarkPerf(b *testing.B) {
	var (
		result  *yolov5face.Result
		service *yolov5face.Service
		imgByte []byte
		file    string
		err     error
	)

	file = os.Args[len(os.Args) - 1]

	service = yolov5face.NewHttpService(
		"http://localhost:8080/predictions/fd1",
		time.Millisecond * time.Duration(50),
		1920,
		1080,
	)
	if !service.IsAlive() {
		b.Error("service is not availabe")
		b.FailNow()
	}

	if imgByte, err = ioutil.ReadFile(file); err != nil {
		b.Error(err)
	}

	test := func () {
		if result, err = service.DetectHeadFromByte(imgByte); err != nil {
			b.Log(err)
		}
		_ = result
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			test()
		}
	})
}
