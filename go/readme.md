# The go API module of yolov5face


Description:
This go module provide API to call face detect remote RPC.


Usage:
import "gitlab.gtvorg.tk/pannixilin/yolov5face/go" in go file
The go/service_test.go provides an coding example for this API.


Test:
This module contains 2 tests and 1 sample data.
The example test:
yolov5face/go/service_test.go
The performance test:
yolov5face/go/test/perf_test.go
The sample image:
yolov5face/go/test/zidane.jpg

Follow below instructions to run each test.

a. service_test.go:
   1. cd yolov5face/go
   2. go test -v

b. yolov5face/go/test/perf_test.go:
   1. cd yolov5face/go/test
   2. go test -bench=Perf -cpu 1,2,3,4,5,6,7,8 -benchtime 30s -args /a/path/to/a/image/file
      for example:
      go test -bench=Perf -cpu 1,2,3,4,5,6,7,8 -benchtime 30s -args zidane.jpg
      The parameter "-cpu 1,2,3,..." luanches different banchmark tests using specified
      parallel excution number.