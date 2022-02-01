package main

import (
	"bufio"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var readIMGs int
var guesNum int

type NN struct {
	config      []int
	innerLayers []hidenLayer
	outLayer    []float64
	lernLayer   []float64
}

func NewANN(config []int) (nn *NN, err error) {
	if len(config) < 3 {
		return nil, errors.New("Not enough arguments")
	}

	nn = &NN{config: config}

	nn.innerLayers = make([]hidenLayer, len(config)-1)
	for i := range nn.innerLayers {
		nn.innerLayers[i] = NewLayer(config[i], config[i+1])
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

	filename := "./export/" + name + "_" + strconv.Itoa(count) + "/nnExport.txt"

	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY, 0644)

	if err != nil {
		return err
	}

	datawriter := bufio.NewWriter(file)

	for i, layer := range n.innerLayers {
		for j, arr := range layer.layerWeights {
			_, _ = datawriter.WriteString("# " + strconv.Itoa(i) + " " + strconv.Itoa(j) + " " + strconv.Itoa(len(arr)) + "\n")
			for _, elem := range arr {
				_, _ = datawriter.WriteString(strconv.FormatFloat(elem, 'f', -1, 64) + " ")
			}
			_, _ = datawriter.WriteString("\n")
		}
	}

	datawriter.Flush()
	file.Close()

	return nil
}

func (n *NN) firstGenerate() error {
	fmt.Println("NN random generated")
	if n == nil || n.innerLayers == nil {
		return errors.New("Bad arguments(random initialise)")
	}

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	for _, layer := range n.innerLayers {
		for _, arr := range layer.layerWeights {
			for i := range arr {
				arr[i] = (randGen.Float64() - 0.5) * 0.5
			}
		}
	}

	return nil
}

func (nn *NN) TrainNN(trainData [][]float64, epochs int) error {

	for i := 0; i < epochs; i++ {
		err := nn.nnGo(trainData, i, true)
		if err != nil {
			return err
		}

		/* exportStr := "epoch_" + strconv.Itoa(i) + "_" + strconv.Itoa(readIMGs) + "_" + strconv.Itoa(guesNum)
		err = nn.nnExport(exportStr)
		if err != nil {
			return err
		} */
	}

	exportStr := "epoch_" + strconv.Itoa(readIMGs) + "_" + strconv.Itoa(guesNum)
	err := nn.nnExport(exportStr)
	if err != nil {
		return err
	}

	return nil
}

func (nn *NN) nnGo(trainData [][]float64, epoch int, mode bool) error {

	readIMGs = 0
	guesNum = 0

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

	fmt.Printf("%.3f ", float32(guesNum)/float32(readIMGs))
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
		err := layer.layerForvard(target)
		if err != nil {
			return err
		}
	}

	return nil
}

func (nn NN) check() []float64 {

	res := make([]float64, len(nn.lernLayer))

	for i := 0; i < len(nn.lernLayer); i++ {
		res[i] = nn.lernLayer[i] - nn.outLayer[i]
	}

	resNum, _ := maxInArray(nn.outLayer)
	targetNum, _ := maxInArray(nn.lernLayer)

	if resNum == targetNum {
		guesNum++
	}

	readIMGs++

	return res
}

func (nn *NN) nnBP(mistArr []float64) {

	curLayer := nn.outLayer

	for i := len(nn.innerLayers) - 1; i >= 0; i-- {

		tmpLayer := make([]float64, len(curLayer))

		for j, e := range curLayer {
			tmpLayer[j] = sigmoidPrime(e)
		}

		mistArr = nn.innerLayers[i].layerBP(mistArr, tmpLayer)
		curLayer = nn.innerLayers[i].layerContent

	}
}

func (nn *NN) loadStoredANN() error {

	folder := "./export/"
	var lastExportFile string

	err := filepath.Walk(folder, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			fmt.Println(err)
			return err
		}
		lastExportFile = path
		return nil
	})

	if err != nil {
		return err
	}

	if folder == lastExportFile {
		return errors.New("No files to import")
	}

	fmt.Println("NN loaded from file: ", lastExportFile)

	file, err := os.Open("./" + lastExportFile)
	if err != nil {
		return err
	}

	defer file.Close()

	scanner := bufio.NewScanner(file)

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
		log.Fatal(err)
	}

	return nil
}
