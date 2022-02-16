package yolov5face

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"time"
)

type Service struct {
	client    *http.Client
	url       string
	maxWidth  int
	maxHeight int
}

func NewHttpService(url string, timeout time.Duration, maxWidth int, maxHeight int) *Service {
	maxWidth, maxHeight = unifyImageWidthHeight(maxWidth, maxHeight)
	return &Service{
		client: &http.Client{
			Timeout: timeout,
		},
		url: url,
		maxWidth: maxWidth,
		maxHeight: maxHeight,
	}
}

func (s *Service) IsAlive() bool {
	var (
		req  *http.Request
		resp *http.Response
		err  error
	)
	if req, err = http.NewRequest(http.MethodPut, s.url, nil); err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	if resp, err = s.client.Do(req); err != nil {
		return false
	} else {
		resp.Body.Close()
		return true
	}
}

func (s *Service) DetectHeadFromFile(file string) (*Result, error) {
	if b, err := ioutil.ReadFile(file); err != nil {
		return nil, err
	} else {
		return s.DetectHeadFromByte(b)
	}
}

func (s *Service) DetectHeadFromByte(b []byte) (*Result, error) {
	r := bytes.NewReader(b)
	return s.DetectHeadFromReader(r)
}

func (s *Service) DetectHeadFromReader(img *bytes.Reader) (*Result, error) {
	var (
		httpResults []HttpResult
		width       int
		height      int
		err         error
	)
	if width, height, err = GetImageSize(img); err != nil {
		return nil, err
	}
	if s.isLargeImage(width, height) {
		return nil, fmt.Errorf("large image is skipped")
	}
	img.Seek(0, io.SeekStart)
	if httpResults, err = s.DetectHead(img); err != nil {
		return nil, err
	}
	return HttpResult2Result(width, height, httpResults), nil
}

func (s *Service) DetectHead(img io.Reader) ([]HttpResult, error) {
	var (
		httpResults []HttpResult
		req         *http.Request
		resp        *http.Response
		body        []byte
		err         error
	)
	if req, err = http.NewRequest(http.MethodPut, s.url, img); err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	if resp, err = s.client.Do(req); err != nil {
		return nil, err
	}
        defer resp.Body.Close()
        if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http error: %s", resp.Status)
	}
	if body, err = ioutil.ReadAll(resp.Body); err != nil {
		return nil, err
	}
	if err = json.Unmarshal(body, &httpResults); err != nil {
		return nil, err
	}
	return httpResults, nil
}

func (s *Service) isLargeImage(width int, height int) bool {
	width, height = unifyImageWidthHeight(width, height)
	if s.maxWidth > 0 && s.maxHeight > 0 {
		return width > s.maxWidth || height > s.maxHeight
	} else {
		return false
	}
}
