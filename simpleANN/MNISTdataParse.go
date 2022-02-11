package main

import (
	"bufio"
	"encoding/binary"
	"os"
)

const (
	trainImg   string = "./data/train-images.idx3-ubyte"
	trainLabel string = "./data/train-labels.idx1-ubyte"
	examImg    string = "./data/t10k-images.idx3-ubyte"
	examLabel  string = "./data/t10k-labels.idx1-ubyte"
)

func GetMNISTdata(train bool) ([][]float64, int, error) {
	var dataPath, labelPath string

	if train {
		dataPath, labelPath = trainImg, trainLabel

	} else {
		dataPath, labelPath = examImg, examLabel
	}

	data, err := getDataAsByteArr(dataPath)
	label, err2 := getDataAsByteArr(labelPath)
	if err != nil {
		return nil, 0, err
	}
	if err2 != nil {
		return nil, 0, err2
	}

	resData, dim, err := getDataSetSlices(data, label)
	if err != nil {
		return nil, 0, err
	}

	return resData, dim, nil
}

func getDataAsByteArr(filename string) ([]byte, error) {

	file, err := os.Open(filename)

	if err != nil {
		return nil, err
	}
	defer file.Close()

	stats, statsErr := file.Stat()
	if statsErr != nil {
		return nil, statsErr
	}

	var size int64 = stats.Size()
	bytes := make([]byte, size)

	bufr := bufio.NewReader(file)
	_, err = bufr.Read(bytes)
	if err != nil {
		return nil, err
	}

	return bytes, nil
}

func getDataSetSlices(data, label []byte) ([][]float64, int, error) {

	var outData [][]float64

	offcet1 := 16
	offcet2 := 8

	var config [4]int
	for i := 0; i < 4; i++ {
		config[i] = int(binary.BigEndian.Uint32(data[i*4 : i*4+4]))
	}

	a := offcet1
	c := offcet2

	for i := 0; i < config[1]; i++ {

		curData := make([]float64, config[2]*config[3])
		targetArr := make([]float64, 10)

		b := a + config[3]*config[2]
		for idx, elem := range data[a:b] {
			curData[idx] = float64(elem)/float64(255) + 0.0001
		}
		a += config[3] * config[2]

		targetArr[int(label[c])] = float64(1)

		c++
		curData = append(curData, targetArr...)

		outData = append(outData, curData)
	}

	return outData, config[2], nil
}
