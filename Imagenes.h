#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <windows.h>
#include <windows.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using std::cin;
using std::cout;
using std::endl;
using namespace cv;
using namespace std;
class Imagenes {
public:
	//Constructor
	Imagenes() {
		imagen = imread(ruta);
		imagenNeg = imread(rutaNeg);
		imagenBrill = imread(rutaBrill);
		imagenContr = imread(rutaContr);
		imagenAuto = imread(rutaAuto);
		imagenGamma = imread(rutaGamma);
		imagenBorroso = imread(rutaKernelBorr);
		imagenProm = imread(rutaKernelProm);
		imagenS = imread(rutaSobel);
		imagenl = imread(rutaLaplace);
		imagenB = imread(rutaBinomial);
		imagenPasaB = imread(rutaPasaB);
		imagenPasaA = imread(rutaPasaA);
		imagenButter = imread(rutaButter);
		imagenGrafico = imread(rutaGrafico);
		imagenButterAl = imread(rutaButterAl);
		imagenHomomo = imread(rutaHomomor);
	}

	//Metodos
	Mat MostImagen(), MostImagenNeg(), MostImagenBrill(), MostImagenContr(), MostImagenAuto(), MostImagenGamma();
	Mat CalcHistograma(Mat src), CalcHistograma2(Mat src2);
	void MostrarHistograma(Mat histogram), MostrarHistograma2(Mat histograma2);
	void SubMenuHistograma();
	int histograma(), histogramaecualizado(), GradienteX(Mat image, int x, int y), GradienteY(Mat image, int x, int y);
	void Convolucion(), FiltroBinomial();
	void FiltroMinimo(), FiltroSobel(), FiltroLaplace();
	void TDF(), PasaBajo(), PasaAlto();
	void ButterWorth(), homomorfico();
	Mat Butterworth_Low_Paass_Filter(Mat& src, float d0, int n), butterworth_lbrf_kernel(Mat& scr, float sigma, int n);

private:
	//atributos
	String ruta = "lena.pgm";
	String rutaNeg = "CargarImagennegativo.pgm";
	String rutaBrill = "CargarImagenBrillo.pgm";
	String rutaContr = "Contraste.pgm";
	String rutaAuto = "ContrasteAutomatico.pgm";
	String rutaGamma = "CoreccionGamma.pgm";
	String rutaKernelBorr = "KernelBorroso.pgm";
	String rutaKernelProm = "KernelIdentidad.pgm";
	String rutaSobel = "FiltroSobel.pgm";
	String rutaLaplace = "FiltroLaplaciano.pgm";
	String rutaBinomial = "FiltroBinomial.pgm";
	String rutaPasaA = "FiltroPasaAlto.pgm";
	String rutaPasaB = "FiltroPasaBajo.pgm";
	String rutaButter = "Butterworth.pgm";
	String rutaGrafico = "grafico.bmp";
	String rutaButterAl = "butterworthAltas.pgm";
	String rutaHomomor = "Homomorfico.pgm";
	Mat imagenButter, imagenGrafico, imagenNeg, imagenBrill, imagenContr, imagenAuto, imagenGamma, imagenProm, imagenBorroso, imagenMinMax, imagenS, imagenl, imagenB, imagenPasaA, imagenPasaB, imagen, imagenButterAl, imagenHomomo;
	Mat hist, hist2;
	double valor;
	int bmpwidth, bmpheight, linebyte;
	unsigned char* pBmpBuf;
	int gx, gy, sob;
	double A = 0.5;
	int f = 7;
	double t[100];
	double output[100];
	int length = 100;
	double phi = CV_PI / 4;
	double result[100];
	double sinusoid[100][100];
};