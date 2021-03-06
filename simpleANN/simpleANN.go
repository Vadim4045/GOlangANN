package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type NN struct {
	alfa        float64
	mu          float64
	readIMGs    int
	guesNum     int
	prevEpoch   int
	config      []int
	innerLayers []hidenLayer
	outLayer    []float64
	lernLayer   []float64
}

func NewANN(simpleAlfa, simpleMu float64, config []int) (nn *NN, err error) {
	if len(config) < 3 {
		return nil, errors.New("Not enough arguments")
	}

	nn = &NN{config: config}

	nn.alfa = simpleAlfa
	nn.mu = simpleMu

	nn.innerLayers = make([]hidenLayer, len(config)-1)
	for i := range nn.innerLayers {
		nn.innerLayers[i] = NewLayer(config[i], config[i+1], nn.alfa)
	}

	nn.outLayer = make([]float64, config[len(config)-1])
	nn.lernLayer = make([]float64, config[len(config)-1])

	err = nn.loadStoredANN()
	if err != nil {
		fmt.Println("Load error", err)
		err = nn.firstGenerate()
		if err != nil {
			fmt.Println("Init error: ", err)
			os.Exit(1)
		}
	}
	return
}

func (n NN) nnExport(name string) error {
	count := 0

	for {
		if err := os.Mkdir("./export/"+name+"_"+strconv.Itoa(count), 0755); os.IsExist(err) {
			count++
		} else {
			break
		}
	}

	filename := "./export/" + name + "_" + strconv.Itoa(count) + "/nn.bin"

	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	for _, layer := range n.innerLayers {
		for _, arr := range layer.layerWeights {
			for _, elem := range arr {
				binary.Write(file, binary.LittleEndian, elem)
			}
		}
	}

	return nil
}

func (nn *NN) firstGenerate() error {
	nn.prevEpoch = 0
	fmt.Println("NN random generated")
	if nn == nil || nn.innerLayers == nil {
		return errors.New("Bad arguments(random initialise)")
	}

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	for _, layer := range nn.innerLayers {
		for _, arr := range layer.layerWeights {
			for i := range arr {
				arr[i] = (randGen.Float64() - 0.5) * 0.1
			}
		}
	}

	return nil
}

func (nn *NN) TrainNN(trainData [][]float64, epochs int, mode bool) error {

	for i := nn.prevEpoch + 1; i <= nn.prevEpoch+epochs; i++ {

		if !myGo {
			return nil
		}

		nn.mu = mu

		err := nn.nnGo(trainData, i, mode)
		if err != nil {
			return err
		}

		annStr := ""
		for _, conf := range nn.config {
			annStr += fmt.Sprintf("%d_", conf)
		}
		exportStr := fmt.Sprintf("%.4d_%s%.5d_%.5d_%.1f_%.6f", i, annStr, nn.readIMGs, nn.guesNum, nn.alfa, nn.mu)

		fmt.Println(exportStr)

		err = nn.nnExport(exportStr)
		if err != nil {
			return err
		}
	}

	return nil
}

func (nn *NN) nnGo(trainData [][]float64, epoch int, mode bool) error {

	nn.readIMGs = 0
	nn.guesNum = 0

	start := time.Now()

	for _, curIMG := range trainData {

		if len(curIMG) != nn.config[0]+len(nn.outLayer) {
			return errors.New("Training error: input data does not match ANN config")
		}

		for idx := range nn.innerLayers[0].layerContent {
			nn.innerLayers[0].layerContent[idx] = curIMG[idx]
		}

		offcet := len(curIMG) - len(nn.lernLayer)
		for i := 0; i < len(nn.lernLayer); i++ {
			nn.lernLayer[i] = curIMG[offcet+i]
		}

		err := nn.nnFP()
		if err != nil {
			return err
		}

		mistArr := nn.check()

		if mode {
			nn.nnBP(mistArr)
		}
	}

	fmt.Printf("%.2f/%.5f/%.4f_on_%v - ", nn.alfa, nn.mu, float32(nn.guesNum)/float32(nn.readIMGs), time.Now().Sub(start))
	return nil
}

func (nn *NN) nnFP() error {
	var target *[]float64

	for l, layer := range nn.innerLayers {
		if l < len(nn.config)-2 {
			target = &nn.innerLayers[l+1].layerContent
		} else {
			target = &nn.outLayer
		}
		err := layer.layerForvard(target, nn.alfa)
		if err != nil {
			return err
		}
	}

	return nil
}

func (nn *NN) check() []float64 {

	res := make([]float64, len(nn.lernLayer))

	for i := 0; i < len(nn.lernLayer); i++ {
		res[i] = nn.lernLayer[i] - nn.outLayer[i]
	}

	resNum, _ := maxInArray(nn.outLayer)
	targetNum, _ := maxInArray(nn.lernLayer)

	if resNum == targetNum {
		nn.guesNum++
	}

	nn.readIMGs++

	return res
}

func (nn *NN) nnBP(mistArr []float64) {

	curLayer := nn.outLayer

	for i := len(nn.innerLayers) - 1; i >= 0; i-- {

		tmpLayer := make([]float64, len(curLayer))

		for j, e := range curLayer {
			wg.Add(1)

			go func(k int, l float64) {
				tmpLayer[k] = sigmoidPrime(l, nn.innerLayers[i].alfa)
				wg.Done()
			}(j, e)
		}

		wg.Wait()

		mistArr = nn.innerLayers[i].layerBP(mistArr, tmpLayer, nn.mu)
		curLayer = nn.innerLayers[i].layerContent

	}
}

func (nn *NN) loadStoredANN() error {

	binExport := true

	folder := "./export/"
	exportFiles := make([]string, 1)

	var paramString string = "_"
	for i := 0; i < len(nn.config); i++ {
		paramString += fmt.Sprintf("%d_", nn.config[i])
	}

	err := filepath.Walk(folder, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			fmt.Println(err)
			return err
		}

		if binExport {
			if strings.Contains(path, paramString) && strings.Contains(path, ".bin") {
				exportFiles = append(exportFiles, path)
			}
		} else {
			if strings.Contains(path, paramString) && strings.Contains(path, ".txt") {
				exportFiles = append(exportFiles, path)
			}
		}

		return nil
	})

	if err != nil {
		return errors.New("Bad path")
	}

	if len(exportFiles) == 1 {
		return errors.New("No suitable files to import")
	}

	lastExportFile := exportFiles[len(exportFiles)-1]

	params := strings.Split(lastExportFile, "_")

	if len(params) < 10 {
		return errors.New("Bad filename length")
	}

	for idx := 0; idx < len(nn.config); idx++ {
		nextCount, err := strconv.Atoi(params[idx+1])
		if err != nil {
			return errors.New("Bad filename params")
		}

		if nn.config[idx] != nextCount {
			return errors.New("This file not for current network parameters")
		}
	}

	fmt.Println("NN loaded from file: ", lastExportFile)

	f, err := os.Open("./" + lastExportFile)
	if err != nil {
		return err
	}
	defer f.Close()

	nn.prevEpoch, err = strconv.Atoi(strings.Split(params[0], "/")[1])
	if err != nil {
		nn.prevEpoch = 0
	}

	if binExport {
		r := bufio.NewReader(f)
		buf := make([]byte, 0, 8)

		for i, _ := range nn.innerLayers {
			for j, _ := range nn.innerLayers[i].layerWeights {
				for k, _ := range nn.innerLayers[i].layerWeights[j] {

					n, err := r.Read(buf[:cap(buf)])
					if err != nil && err != io.EOF {
						return err
					}

					buff := bytes.NewReader(buf[:n])

					err = binary.Read(buff, binary.LittleEndian, &nn.innerLayers[i].layerWeights[j][k])
					if err != nil {
						return err
					}
				}
			}
		}
	} else {
		scanner := bufio.NewScanner(f)

		flag := true
		var layer, slice, length, count int

		for scanner.Scan() {

			if flag {
				config := strings.Split(strings.TrimSpace(scanner.Text()), " ")
				if config[0] == "#" {

					flag = false
					layer, err = strconv.Atoi(config[1])
					if err != nil {
						return err
					}
					slice, err = strconv.Atoi(config[2])
					if err != nil {
						return err
					}
					length, err = strconv.Atoi(config[3])
					if err != nil {
						return err
					}
					count = 0
				} else {
					return errors.New("Bad export file")
				}
			} else {
				line := strings.Split(strings.TrimSpace(scanner.Text()), " ")
				for i := 0; i < length; i++ {
					tmp, err := strconv.ParseFloat(line[i], 64)
					if err != nil {
						return err
					}
					nn.innerLayers[layer].layerWeights[slice][i] = tmp
					count++
					if count == length {
						flag = true
					}

				}
			}

		}

		if err := scanner.Err(); err != nil {
			return err
		}
	}

	return nil
}
