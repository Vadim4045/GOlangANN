package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
)

const (
	trainImg   string = "./data/train-images.idx3-ubyte"
	trainLabel string = "./data/train-labels.idx1-ubyte"
	examImg    string = "./data/t10k-images.idx3-ubyte"
	examLabel  string = "./data/t10k-labels.idx1-ubyte"
)

func GetMNISTdata(train bool, mask int) ([][]float64, int, error) {
	var dataPath, labelPath string

	if train {
		dataPath, labelPath = trainImg, trainLabel

	} else {
		dataPath, labelPath = examImg, examLabel
	}

	data, err := getDataAsByteArr(dataPath)
	label, err2 := getDataAsByteArr(labelPath)
	if err != nil {
		//fmt.Println("Error read IMGset: ", err)
		return nil, 0, err
	}
	if err2 != nil {
		//fmt.Println("Error read LABELset: ", err2)
		return nil, 0, err2
	}

	resData, dim, err := getDataSetSlices(data, label, mask)
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

func getDataSetSlices(data, label []byte, mask int) ([][]float64, int, error) {

	var outData [][]float64
	var dim int
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
			curData[idx] = float64(elem)/float64(255) + 0.01
		}
		a += config[3] * config[2]

		resized, dimensios := resizeMask(curData, mask)
		if dim == 0 {
			dim = dimensios
		} else if dim != dimensios {
			return nil, 0, errors.New("Bad resizing")
		}

		targetArr[int(label[c])] = float64(1)

		c++
		resized = append(resized, targetArr...)
		outData = append(outData, resized)
	}
	return outData, dim, nil
}

func resizeMask(data []float64, mask int) ([]float64, int) {
	dim := int(math.Sqrt(float64(len(data))))

	if mask > 0 {
		var resArr []float64

		for i := 0; i < dim-mask; i += mask {
			for j := 0; j < dim-mask; j += mask {
				var res float64 = 0
				for m := 0; m < mask; m++ {
					for n := 0; n < mask; n++ {
						if i+m < dim && j+n < dim {
							if data[dim*(i+m)+j+n] > res {
								res = data[dim*(i+m)+j+n]
							}

						}
					}
				}
				/* if res > 0 {
					fmt.Print("00")
				} else {
					fmt.Print("  ")
				}
				*/
				resArr = append(resArr, res)
			}
			fmt.Println("")
		}

		return resArr, dim
	}
	return data, dim
}
