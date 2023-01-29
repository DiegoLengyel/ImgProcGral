#pragma once
#include "Imagenes.h"

int Imagenes::histograma() {
    Mat imagen = imread("C:\\Users\\miele\\Desktop\\lena.pgm");

    // verificar si la imagen ha sida cargada correctamente 
    if (imagen.empty())
    {
        printf("No image data \n");
        return -1;
    }
    //cvtColor(imagen, imagen, COLOR_BGR2GRAY);
    vector<Mat> bgr_planes;
    split(imagen, bgr_planes);
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }
    imshow("Original", imagen);
    imshow("Histograma Original", histImage);
    waitKey(0);
    return 0;
}

int Imagenes::histogramaecualizado() {
    Mat image = imread("C:\\Users\\miele\\Desktop\\lena.pgm");
    if (image.empty())
    {
        printf("No se encontro la imagen.");
        return -1;
    }
    cvtColor(imagen, imagen, COLOR_BGR2GRAY);
    Mat dst;
    equalizeHist(imagen, dst);
    vector<Mat> bgr_planes;
    split(dst, bgr_planes);
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage2(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    normalize(b_hist, b_hist, 0, histImage2.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage2.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage2.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(histImage2, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);

    }
    imshow("Imagen Ecualizada", dst);
    imshow("Histograma Ecualizado", histImage2);
    histograma();
    waitKey();
    return 0;
}

Mat Imagenes::MostImagen() {
    Mat imagenmost(imagen.rows, imagen.cols, CV_8UC3);
    for (int i = 0; i < imagen.rows; i++) {
        for (int j = 0; j < imagen.cols; j++) {
            Vec3b pixel = imagen.at<Vec3b>(i, j);
            int R = pixel[0];
            int G = pixel[0];
            int B = pixel[0];
            if (B > 255) { B = 255; }
            else if (G > 255) { G = 255; }
            else if (R > 255) { R = 255; }
            else if (R < 0) { R = 0; }
            pixel[0] = (uchar)B;
            pixel[1] = (uchar)G;
            pixel[2] = (uchar)R;
            imagenmost.at<Vec3b>(i, j) = pixel;
        }
    }
    imshow("Original", imagenmost);
    waitKey(0);
    return imagenmost;
}

Mat Imagenes::MostImagenNeg() {
    Mat imagenmost(imagenNeg.rows, imagenNeg.cols, CV_8UC3);
    for (int i = 0; i < imagenNeg.rows; i++) {
        for (int j = 0; j < imagenNeg.cols; j++) {
            Vec3b pixel = imagenNeg.at<Vec3b>(i, j);
            int R = pixel[0];
            int G = pixel[0];
            int B = pixel[0];
            if (B > 255) { B = 255; }
            else if (G > 255) { G = 255; }
            else if (R > 255) { R = 255; }
            else if (R < 0) { R = 0; }
            pixel[0] = (uchar)B;
            pixel[1] = (uchar)G;
            pixel[2] = (uchar)R;
            imagenmost.at<Vec3b>(i, j) = pixel;
        }
    }
    imshow("Negativo", imagenmost);
    waitKey(0);
    return imagenmost;
}

Mat Imagenes::MostImagenBrill() {
    imshow("Brillo", imagenBrill);
    waitKey(0);
    return imagenBrill;
}

Mat Imagenes::MostImagenContr() {
    imshow("Contraste", imagenContr);
    waitKey(0);
    return imagenContr;
}

Mat Imagenes::MostImagenAuto() {
    imshow("Contraste Automatico", imagenAuto);
    waitKey(0);
    return imagenAuto;
}

Mat Imagenes::MostImagenGamma() {
    imshow("Correccion Gamma", imagenGamma);
    waitKey(0);
    return imagenGamma;
}

void Imagenes::Convolucion() {
    Mat imagenConv;
    Mat kernel1 = (Mat_<double>(3, 3) << 0, 0, 0, 0, 1, 0, 0, 0, 0);
    //filter2D(imagen, imagenProm, CV_16U / CV_16S, kernel1, Point(-1, -1), 0, 4);
    GaussianBlur(imagen, imagenProm, Size(11, 11), 0, 0, 4);
    //Laplacian(imagen, imagenProm, CV_16U / CV_16S, 1, 1, 0, 4);
    waitKey();
    Mat kernel2 = Mat::ones(5, 5, CV_64F);
    kernel2 = kernel2 / 25;
    //Mat kernel2 = (Mat_<double>(3, 3) << 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9);
    filter2D(imagen, imagenBorroso, -1, kernel2, Point(-1, -1), 0, 4);
    imshow("Original", imagen);
    //imshow("Kernel Borroso", imagenBorroso);
    //imshow("Kerner Promedio", imagenProm);
    imshow("filtro por convolucion", imagenBorroso);
    imshow("filtro por corelacion", imagenProm);
    waitKey(0);
    destroyAllWindows();
}

void Imagenes::FiltroMinimo() {
    Mat Minimo(imagen.rows, imagen.cols, CV_8UC1);
    float Matriz[9];
    /*Laplacian(imagen, imagenMinMax, CV_16U / CV_16S, 1, 1, 0, 4);
    imshow("filtro por Minimo Maximo", imagenMinMax);
    waitKey(0);
    destroyAllWindows();*/
    for (int y = 1; y < imagen.rows; y++)
    {
        for (int x = 1; x < imagen.cols; x++) {
            Matriz[4] = Minimo.at<uchar>(y, x) = min(imagen.at<Vec3b>(y, x)[0], min(imagen.at<Vec3b>(y, x)[1], imagen.at<Vec3b>(y, x)[2]));
            Minimo.at<uchar>(y, x) = Matriz[4];

        }
    }
    imshow("Filtro Minimo", Minimo);
    imshow("Original", imagen);
    waitKey(0);
}

void Imagenes::FiltroBinomial() {
    // int bi[] = {1,4,6,4,1};//triangulo de pasacal
    for (int t = 0; t < 5; t++)
    {
        for (int i = 0; i < t + 1; i++)
        {
            int P;
            P = (tgamma(t + 1)) / ((tgamma(i + 1)) * tgamma((t - i) + 1)); //triangulo de pascal hasta 4
            for (int y = 1; y < imagen.rows - 1; y++) {
                for (int x = 1; x < imagen.cols - 1; x++) {
                    int pnt = imagen.at<uchar>(y, x);
                    pnt = pnt * (P);//forma 1
                    pnt = pnt > 255 ? 255 : pnt;
                    pnt = pnt < 0 ? 0 : pnt;
                    imagen.at<uchar>(y, x) = saturate_cast<uchar>(pnt);
                }
            }
        }
    }
    imshow("Filtro Binomial", imagenB);
    waitKey(0);
}

#pragma region Sobel
int Imagenes::GradienteX(Mat image, int x, int y)
{
    return image.at<uchar>(y - 1, x - 1) +
        image.at<uchar>(y, x - 1) +
        image.at<uchar>(y + 1, x - 1) -
        image.at<uchar>(y - 1, x + 1) -
        image.at<uchar>(y, x + 1) -
        image.at<uchar>(y + 1, x + 1);
}

int Imagenes::GradienteY(Mat image, int x, int y)
{
    return image.at<uchar>(y - 1, x - 1) +
        image.at<uchar>(y - 1, x) +
        image.at<uchar>(y - 1, x + 1) -
        image.at<uchar>(y + 1, x - 1) -
        image.at<uchar>(y + 1, x) -
        image.at<uchar>(y + 1, x + 1);
}

void Imagenes::FiltroSobel() {
    for (int y = 0; y < imagen.rows; y++) {
        for (int x = 0; x < imagen.cols; x++) {
            imagen.at<uchar>(y, x) = 0.0;
        }
    }

    for (int y = 1; y < imagen.rows - 1; y++) {
        for (int x = 1; x < imagen.cols - 1; x++) {
            gx = GradienteX(imagen, x, y);
            gy = GradienteY(imagen, x, y);
            sob = abs(gx) + abs(gy);
            sob = sob > 255 ? 255 : sob;
            sob = sob < 0 ? 0 : sob;
            imagen.at<uchar>(y, x) = sob;
        }
    }
    putText(imagen, "Filto Sobel", Point(0, 25), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
    imshow("Filtro Sobel", imagenS);
    waitKey(0);
}
#pragma endregion

void Imagenes::FiltroLaplace() {
    for (int y = 1; y < imagen.rows - 1; y++) {
        for (int x = 1; x < imagen.cols - 1; x++) {
            int sum = imagen.at<uchar>(y - 1, x)
                + imagen.at<uchar>(y + 1, x)
                + imagen.at<uchar>(y, x - 1)
                + imagen.at<uchar>(y, x + 1)
                - 4 * imagen.at<uchar>(y, x);
            imagen.at<uchar>(y, x) = saturate_cast<uchar>(sum);
        }
    }
    putText(imagen, "Filtro Laplaciano", Point(0, 25), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
    imshow("Filtro Laplaciano", imagenl);
    waitKey(0);
}

#pragma region Fourier
Mat computeTDF(Mat image) {
    Mat padded;
    int m = getOptimalDFTSize(image.rows);
    int n = getOptimalDFTSize(image.cols); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complex;
    merge(planes, 2, complex);
    dft(complex, complex, DFT_COMPLEX_OUTPUT);
    return complex;
}

void fftShift(Mat magI) {

    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));
    Mat q1(magI, Rect(cx, 0, cx, cy));
    Mat q2(magI, Rect(0, cy, cx, cy));
    Mat q3(magI, Rect(cx, cy, cx, cy));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void magnitudeSpectrum(Mat complex) {
    Mat magI;
    Mat planes[] = {
        Mat::zeros(complex.size(), CV_32F),
        Mat::zeros(complex.size(), CV_32F)
    };
    split(complex, planes);
    magnitude(planes[0], planes[1], magI);
    magI += Scalar::all(1);
    log(magI, magI);
    fftShift(magI);
    normalize(magI, magI, 1, 0, NORM_INF);
    imshow("Magnitude Spectrum", magI);
}

void Imagenes::TDF() {
    for (int i = 0; i < length; i++)
    {
        t[i] = i / 100.0;
    }

    for (int i = 0; i < length; ++i)
    {
        result[i] = A * cos(2 * CV_PI * f * t[i] + phi);
    }

    Mat img(1, 100, CV_8UC1, result), output;
    cv::normalize(img, output, 0, 1, NORM_MINMAX);

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length; j++)
        {
            sinusoid[i][j] = result[j] * length;
        }
    }
    Mat Sinusoid1(100, 100, CV_64F, sinusoid);

    imshow("Imagen Original", Sinusoid1);

    Mat dft_1 = computeTDF(Sinusoid1);
    magnitudeSpectrum(dft_1);
    waitKey(0);

    Mat Sinusoid2 = Sinusoid1.t();
    imshow("Imagen Original", Sinusoid2);

    Mat dft_2;
    dft_2 = computeTDF(Sinusoid2);
    magnitudeSpectrum(dft_2);
    waitKey(0);

    // Linear combination of 2, 2d signal
    Mat combine_signal;
    cv::add(Sinusoid1, Sinusoid2, combine_signal);
    imshow("Imagen Original", combine_signal);

    cv::Mat dft_1_2 = computeTDF(combine_signal);
    magnitudeSpectrum(dft_1_2);
    waitKey(0);
}
#pragma endregion

void Imagenes::PasaBajo() {
    int op; //No puede ser muy grande (usar numeros pequeños)
    Scalar intensidad1 = 0;
    cout << "Ingresar el tamaño de la Mascara" << endl;
    cin >> op;
    for (int i = 0; i < imagen.rows - op; i++)
    {
        for (int j = 0; j < imagen.cols - op; j++)
        {
            Scalar intensidad2; //Esto ayuda a filtrar la frecuencia
            for (int k = 0; k < op; k++)
            {
                for (int l = 0; l < op; l++)
                {
                    intensidad1 = imagenPasaB.at<uint8_t>(i + k, j + l);
                    intensidad2.val[0] += intensidad1.val[0];
                }
            }
            imagenPasaB.at<uchar>(i + (op - 1) / 2, j + (op - 1) / 2) = intensidad2.val[0] / (op * op);
        }
    }
    putText(imagenPasaB, "Filtro Pasa Bajo", Point(0, 25), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
    imshow("Filtro PasaBajo", imagenPasaB);
    waitKey(0);
}

void Imagenes::PasaAlto() {
    int op;
    Scalar intensidad1 = 0;
    cout << "Ingresar el tamaño de la Mascara" << endl;
    cin >> op;

    for (int i = 0; i < imagen.rows - op; i++)
    {
        for (int j = 0; j < imagen.cols - op; j++)
        {
            Scalar intensidad2 = 0;
            for (int k = 0; k < op; k++)
            {
                for (int l = 0; l < op; l++)
                {
                    intensidad1 = imagenPasaA.at<uchar>(i + k, j + l);
                    if ((k == (op - 1) / 2) && (l == (op - 1) / 2))
                    {
                        intensidad2.val[0] += (op * op - 1) * intensidad1.val[0];
                    }
                    else
                    {
                        intensidad2.val[0] += (-1) * intensidad1.val[0];
                    }
                }
            }
            imagen.at<uchar>(i + (op - 1) / 2, j + (op - 1) / 2) = intensidad2.val[0] / (op * op);
        }
    }
    putText(imagenPasaA, "Filtro Pasa Alto", Point(0, 25), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
    imshow("Filtro Pasa Alto", imagenPasaA);
    waitKey(0);
}

#pragma region ButterWorth
Mat freqfilt(Mat& scr, Mat& blur)
{

    Mat plane[] = { scr, Mat::zeros(scr.size(), CV_32FC1) };
    Mat complexIm;
    merge(plane, 2, complexIm);
    dft(complexIm, complexIm);
    split(complexIm, plane);
    int cx = plane[0].cols / 2; int cy = plane[0].rows / 2;
    Mat part1_r(plane[0], Rect(0, 0, cx, cy));
    Mat part2_r(plane[0], Rect(cx, 0, cx, cy));
    Mat part3_r(plane[0], Rect(0, cy, cx, cy));
    Mat part4_r(plane[0], Rect(cx, cy, cx, cy));

    Mat temp;
    part1_r.copyTo(temp);
    part4_r.copyTo(part1_r);
    temp.copyTo(part4_r);

    part2_r.copyTo(temp);
    part3_r.copyTo(part2_r);
    temp.copyTo(part3_r);

    Mat part1_i(plane[1], Rect(0, 0, cx, cy));
    Mat part2_i(plane[1], Rect(cx, 0, cx, cy));
    Mat part3_i(plane[1], Rect(0, cy, cx, cy));
    Mat part4_i(plane[1], Rect(cx, cy, cx, cy));

    part1_i.copyTo(temp);
    part4_i.copyTo(part1_i);
    temp.copyTo(part4_i);

    part2_i.copyTo(temp);
    part3_i.copyTo(part2_i);
    temp.copyTo(part3_i);


    Mat blur_r, blur_i, BLUR;
    multiply(plane[0], blur, blur_r);
    multiply(plane[1], blur, blur_i);
    Mat plane1[] = { blur_r, blur_i };
    merge(plane1, 2, BLUR);


    magnitude(plane[0], plane[1], plane[0]);
    plane[0] += Scalar::all(1);
    normalize(plane[0], plane[0], 1, 0, NORM_MINMAX);

    idft(BLUR, BLUR);
    split(BLUR, plane);
    magnitude(plane[0], plane[1], plane[0]);
    normalize(plane[0], plane[0], 1, 0, NORM_MINMAX);
    return plane[0];
}

void Imagenes::ButterWorth() {
    imshow("Gráfica ButterWorth PasaBajos", imagenGrafico);
    imshow("Filtro ButterWorth PasaBajos", imagenButter);
    imshow("Filtro ButterWorth PasaAltos", imagenButterAl);
    waitKey(0);
}
Mat Imagenes::Butterworth_Low_Paass_Filter(Mat& src, float d0, int n) {
    int M = getOptimalDFTSize(src.rows);
    int N = getOptimalDFTSize(src.cols);
    Mat padded;
    copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols, BORDER_CONSTANT, Scalar::all(0));
    padded.convertTo(padded, CV_32FC1);
    Mat butterworth_kernel = butterworth_lbrf_kernel(padded, d0, n);
    Mat result = freqfilt(padded, butterworth_kernel);    
    return result;
}

Mat Imagenes::butterworth_lbrf_kernel(Mat& scr, float sigma, int n) {
    Mat clon;
    int D0 = 5;
    n = 2;
    clon = imagen.clone();
    for (int i = 0; i < clon.rows; i++) {
        for (int j = 0; j < clon.cols; j++) {
            double d = sqrt(pow((i - clon.rows / 2), 2) + pow((j - clon.cols / 2), 2));//numerator, calculated pow must be float type
            clon.at<float>(i, j) = 1.0 / (1 + pow(d / D0, 2 * n));
        }
    }
    string name = "Butterworth D0=" + std::to_string(sigma) + "n=" + std::to_string(n);
    imshow(name, clon);

    return clon;
}

#pragma endregion

#pragma region Homomórfico
Mat image_add_border(Mat& src)
{
    int w = 2 * src.cols;
    int h = 2 * src.rows;
    std::cout << "src: " << src.cols << "*" << src.rows << std::endl;

    Mat padded;
    copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols,
    BORDER_CONSTANT, Scalar::all(0));
    cout << "opt: " << padded.cols << "*" << padded.rows << std::endl;
    return padded;
}

void center_transform(Mat& src)
{
    for (int i = 0; i < src.rows; i++) {
        float* p = src.ptr<float>(i);
        for (int j = 0; j < src.cols; j++) {
            p[j] = p[j] * pow(-1, i + j);
        }
    }
}

void zero_to_center(cv::Mat& freq_plane)
{
    int cx = freq_plane.cols / 2; int cy = freq_plane.rows / 2;
    Mat part1_r(freq_plane, cv::Rect(0, 0, cx, cy));
    Mat part2_r(freq_plane, cv::Rect(cx, 0, cx, cy));
    Mat part3_r(freq_plane, cv::Rect(0, cy, cx, cy));
    Mat part4_r(freq_plane, cv::Rect(cx, cy, cx, cy));
    Mat tmp;
    part1_r.copyTo(tmp); 
    part4_r.copyTo(part1_r);
    tmp.copyTo(part4_r);
    part2_r.copyTo(tmp);
    part3_r.copyTo(part2_r);
    tmp.copyTo(part3_r);
}


void show_spectrum(Mat& complexI)
{
    Mat temp[] = { Mat::zeros(complexI.size(),CV_32FC1),
                      Mat::zeros(complexI.size(),CV_32FC1) };
    
    split(complexI, temp);
    Mat aa;
    magnitude(temp[0], temp[1], aa);
    
    divide(aa, aa.cols * aa.rows, aa);
    imshow("src_img_spectrum", aa);
}

Mat frequency_filter(cv::Mat& padded, cv::Mat& blur)
{
    Mat plane[] = { padded, cv::Mat::zeros(padded.size(), CV_32FC1) };
    Mat complexIm;
    merge(plane, 2, complexIm);
    dft(complexIm, complexIm);
    show_spectrum(complexIm);
    Mat dst_plane[2];
    multiply(complexIm, blur, complexIm);
    idft(complexIm, complexIm, DFT_INVERSE);
    split(complexIm, dst_plane);    
    center_transform(dst_plane[0]);
    return dst_plane[0];
}

Mat gaussian_homo_kernel(Mat& scr, float rh, float rl, float c, float D0)
{
    Mat gaussian_high_pass(scr.size(), CV_32FC2);
    int row_num = scr.rows;
    int col_num = scr.cols;
    float r = rh - rl;
    float d0 = 2 * D0 * D0;
    for (int i = 0; i < row_num; i++) {
        float* p = gaussian_high_pass.ptr<float>(i);
        for (int j = 0; j < col_num; j++) {
            float d = pow((i - row_num / 2), 2) + pow((j - col_num / 2), 2);
            p[2 * j] = r * (1 - expf(-1 * c * (d / d0))) + rl;
            p[2 * j + 1] = r * (1 - expf(-1 * c * (d / d0))) + rl;
        }
    }

    Mat temp[] = { Mat::zeros(scr.size(), CV_32FC1),
                       Mat::zeros(scr.size(), CV_32FC1) };
    split(gaussian_high_pass, temp);
    string name = "Filtro d0 =" + std::to_string(D0);
    Mat show;
    normalize(temp[0], show, 1, 0, NORM_MINMAX);
    imshow(name, show);
    return gaussian_high_pass;
}

Mat homofilter(Mat image_in, float rh, float rl, float c, float D0)
{
    image_in.convertTo(image_in, CV_32FC1);
    log(image_in + 1, image_in);
    Mat padded = image_add_border(image_in);
    center_transform(padded);

    Mat blur = gaussian_homo_kernel(padded, rh, rl, c, D0);
    Mat dst = frequency_filter(padded, blur);
    normalize(dst, dst, 5, 0, NORM_MINMAX); 

    exp(dst, dst);
    dst = dst - 1;
    return dst;
}

void Imagenes::homomorfico() {
    imshow("Filtro Homomórfico", imagenHomomo);
    waitKey(0);
}
#pragma endregion

#pragma region Histograma
Mat Imagenes::CalcHistograma(Mat src) {
    hist = Mat::zeros(256, 1, CV_32F);
    //Recibimos una imagen donde la funcion zeros inicializa los valores de toma de intensidad
    //Tomando Filas Columnas y un tipo de canal en este caso RGB de 0 a 1
    src.convertTo(src, CV_32F);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            valor = src.at<float>(i, j); //Que valor de intensidad guardamos para cada iteracion
            //Accedemos a cada valor de pixel de forma indivudual y tenemos en cuenta su tipo
            hist.at<float>(valor) = hist.at<float>(valor) + 1; // h[rk]
        }
    }
    return hist; //Retornamos como tal los calculos
}

void Imagenes::MostrarHistograma(Mat histograma) {
    //Creamos una imagen de fondo y forma en ventana
    Mat histogrammost(500, 812, CV_8UC3, Scalar(0, 0, 0)); //Scalar es la funcion de color
    Mat norml_histograma;
    //Normalizar el histograma así como en la estadística
    normalize(histograma, norml_histograma, 0, 400, NORM_MINMAX, -1, Mat());
    for (int x = 0; x < 256; x++) {
        rectangle(histogrammost, Point(3 * x, histogrammost.rows - norml_histograma.at<float>(x)), //rk ... L-1
            Point(3 * (x + 1), histogrammost.rows), Scalar(0, 255, 150)); //# repeticiones por intensidad 3* factor de anchura
    }
    imshow("Histograma", histogrammost);
}

Mat Imagenes::CalcHistograma2(Mat src2) {
    hist2 = Mat::zeros(256, 1, CV_32F);
    //Recibimos una imagen donde la funcion zeros inicializa los valores de toma de intensidad
    //Tomando Filas Columnas y un tipo de canal en este caso RGB de 0 a 1
    src2.convertTo(src2, CV_32F);
    for (int i = 0; i < src2.rows; i++) {
        for (int j = 0; j < src2.cols; j++) {
            valor = src2.at<float>(i, j); //Que valor de intensidad guardamos para cada iteracion
            //Accedemos a cada valor de pixel de forma indivudual y tenemos en cuenta su tipo
            hist2.at<float>(valor) = hist2.at<float>(valor) + 1; // h[rk]
        }
    }
    return hist2; //Retornamos como tal los calculos
}

void Imagenes::MostrarHistograma2(Mat histograma2) {
    //Creamos una imagen de fondo y forma en ventana
    Mat histogrammost2(500, 812, CV_8UC3, Scalar(0, 0, 0)); //Scalar es la funcion de color
    Mat norml_histograma2;
    //Normalizar el histograma así como en la estadística
    normalize(histograma2, norml_histograma2, 0, 400, NORM_MINMAX, -1, Mat());
    for (int x = 0; x < 256; x++) {
        rectangle(histogrammost2, Point(3 * x, histogrammost2.rows - norml_histograma2.at<float>(x)), //rk ... L-1
            Point(3 * (x + 1), histogrammost2.rows), Scalar(255, 0, 150)); //# repeticiones por intensidad 3* factor de anchura
    }
    imshow("Histograma normal", histogrammost2);
}



void Imagenes::SubMenuHistograma() {
    int op;
    do
    {
        cout << endl << endl;
        cout << "Bienvenido al menú de los histogramas" << endl;
        cout << "1. Histograma  - imagen original" << endl << endl;
        cout << "2. Histograma - negativo" << endl << endl;
        cout << "3. Histograma - brillo" << endl << endl;
        cout << "4. Histograma - contraste" << endl << endl;
        cout << "5. Histograma - equalizado" << endl << endl;
        cout << "6. Histograma - correccion gamma" << endl << endl;
        cout << "7. Histograma - filtro minimo (soon tm)" << endl << endl;
        cout << "8. Histograma - filtro maximo (soon tm)" << endl << endl;
        cout << "9. Histograma - convolución (soon tm)" << endl << endl;
        cout << "10. Histograma - convolución (soon tm)" << endl << endl;
        cout << "11. Histograma - Sobel" << endl << endl;
        cout << "12. Histograma - Laplaciano" << endl << endl;
        cout << "13. Histograma - Binomial" << endl << endl;
        cout << "Presione otra tecla para salir del programa." << endl << endl;
        cin >> op;
        switch (op) {
        case 1:
            CalcHistograma2(imagen);
            CalcHistograma(imagenNeg);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 2:
            CalcHistograma2(imagen);
            CalcHistograma(imagenNeg);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 3:
            CalcHistograma2(imagen);
            CalcHistograma(imagenBrill);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 4:
            CalcHistograma2(imagen);
            CalcHistograma(imagenContr);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 5:
            CalcHistograma2(imagen);
            CalcHistograma(imagenAuto);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 6:
            CalcHistograma2(imagen);
            CalcHistograma(imagenGamma);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 7:
            break;
        case 8:
            break;
        case 9:
            CalcHistograma2(imagen);
            CalcHistograma(imagenProm);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 10:
            CalcHistograma2(imagen);
            CalcHistograma(imagenBorroso);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 11:
            CalcHistograma2(imagen);
            CalcHistograma(imagenS);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 12:
            CalcHistograma2(imagen);
            CalcHistograma(imagenl);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        case 13:
            CalcHistograma2(imagen);
            CalcHistograma(imagenB);
            MostrarHistograma2(hist2);
            MostrarHistograma(hist);
            waitKey(0);
            break;
        default:
            exit(0);
            break;
        }
    } while (op < 10);

}
#pragma endregion