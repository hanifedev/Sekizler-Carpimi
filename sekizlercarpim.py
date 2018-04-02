import math

def main():
    w1 = [[2.5, 3.17],
          [1.15, 3.73]]
    w2 = [3.27, 2.50]
    b1 = [[3.72],
          [1.16]]
    b2 = [3.24]
    o_katsayisi = 0.6
    m_katsayisi = 0.3
    x1, x2, bias = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0], 0.8, 1
    bm = [8.0,16.0,24.0,32.0,40.0,48.0,56.0,64.0,72.0,80.0]
    adm = [[0, 0], [0, 0], [0], [0]]
    bias_adm = [[0.0, 0.0], [0.0]]
    toplam_hata=0.0

    for i in range(0,10,1):
        x1[i]/=10.0
        bm[i]/=100.0

    for j in range(1,20,1):
        for i in range(0,7, 1):
            e1 = net_girdi(x1[i], w1[0][0], x2, w1[1][0], bias, b1[0][0])
            print("e1 : " , e1)
            e2 = net_girdi(x1[i], w1[0][1], x2, w1[1][1], bias, b1[1][0])
            print("e2 : " , e2)

            f1 = sigmoid_fonksiyonu(e1)
            print("f1 : ", f1)

            f2 = sigmoid_fonksiyonu(e2)
            print("f2 : ", f2)
            e3 = net_girdi(f1, w2[0], f2, w2[1], bias, b2[0])
            print("e3 : ", e3)

            f3 = sigmoid_fonksiyonu(e3)
            print("f3 : ", f3)
            agin_hatasi = hata_hesapla(bm[i], f3)
            print("hata : ", agin_hatasi)

            print("***************** AGIRLIKLAR GUNCELLENIYOR... *************************")
            print("***************** CIKTI KATMANI AGIRLIKLAR GUNCELLENIYOR... *************************")

            aj1m1 = yeni_agirlik(w2[0], o_katsayisi, m_katsayisi, f3, agin_hatasi, f1, adm[2][0])
            print("yeni agirlik aj1m1 ",aj1m1)

            aj2m1 = yeni_agirlik(w2[1], o_katsayisi, m_katsayisi, f3, agin_hatasi, f2, adm[3][0])
            print("yeni agirlik aj2m1 ", aj2m1)
            bj1m1 = yeni_bias_degeri(b2[0], o_katsayisi, m_katsayisi, agin_hatasi, bias_adm[1][0])
            print("bias bj1m1",bj1m1)
            print("***************** GIRDI KATMANI AGIRLIKLAR GUNCELLENIYOR... *************************")
            ak1j1 = girdi_katmani_yeni_agirlik(w1[0][0], o_katsayisi, m_katsayisi, f3, agin_hatasi, x1[i], w2[0], adm[0][0])
            print("girdi katamni ak1j1",ak1j1)
            ak1j2 = girdi_katmani_yeni_agirlik(w1[0][1], o_katsayisi, m_katsayisi, f3, agin_hatasi, x1[i], w2[1], adm[0][1])
            print("girdi katamni ak1j2", ak1j2)

            ak2j1 = girdi_katmani_yeni_agirlik(w1[1][0], o_katsayisi, m_katsayisi, f3, agin_hatasi, x1[i], w2[0], adm[1][0])
            print("girdi katamni ak2j1", ak2j1)

            ak2j2 = girdi_katmani_yeni_agirlik(w1[1][1], o_katsayisi, m_katsayisi, f3, agin_hatasi, x1[i], w2[1], adm[1][1])
            print("girdi katamni ak2j2", ak2j2)

            bj2m1 = yeni_bias_degeri(b1[0][0], o_katsayisi, m_katsayisi, agin_hatasi, bias_adm[0][0])
            print("bias bj2m1", bj2m1)

            bj2m2 = yeni_bias_degeri(b1[1][0], o_katsayisi, m_katsayisi, agin_hatasi, bias_adm[0][1])
            print("bias bj2m2", bj2m2)
            w2[0]=aj1m1
            w2[1]=aj2m1
            b2[0]=bj1m1
            w1[0][0]=ak1j1
            w1[0][1]=ak1j2
            w1[1][0]=ak2j1
            w1[1][1]=ak2j2
            b1[0][0]=bj2m1
            b1[1][0]=bj2m2
            adm[0][0]=agirlik_degisim_miktari(o_katsayisi, m_katsayisi, f3, agin_hatasi,x1[i], adm[0][0])
            adm[0][1]=agirlik_degisim_miktari(o_katsayisi, m_katsayisi, f3, agin_hatasi,x2, adm[0][1])
            adm[1][0]=agirlik_degisim_miktari(o_katsayisi, m_katsayisi, f3, agin_hatasi,x1[i], adm[1][0])
            adm[1][1]=agirlik_degisim_miktari(o_katsayisi, m_katsayisi, f3, agin_hatasi,x2, adm[1][1])
            print("********************* yeni agirliklar ****************")
            print(" w2[0]",w2[0])
            print(" w2[1]",w2[1])
            print(" b2[0]", b2[0])
            print(" w1[0][0]", w1[0][0])
            print(" w1[0][1]", w1[0][1])
            print(" w1[1][0]", w1[1][0])
            print(" b1[0][0]", b1[0][0])
            print(" b1[1][0]", b1[1][0])
            print("********************",i+1,". iterasyon")
        toplam_hata += math.sqrt(agin_hatasi ** 2)
        print("toplam hata", toplam_hata)
        if (toplam_hata >= 5.0):
            break

        print("********************", j, ". epoch tamamlandi")

    print("*********************** TEST VERILERI **********************")
    for m in range(7,10,1):
        e1 = net_girdi(x1[m], w1[0][0], x2, w1[1][0], bias, b1[0][0])
        e2 = net_girdi(x1[m], w1[0][1], x2, w1[1][1], bias, b1[1][0])
        f1 = sigmoid_fonksiyonu(e1)
        f2 = sigmoid_fonksiyonu(e2)
        e3 = net_girdi(f1, w2[0], f2, w2[1], bias, b2[0])
        f3 = sigmoid_fonksiyonu(e3)
        print(x1[m]*10," x "," 8.0 =" , f3*100)
        agin_hatasi = hata_hesapla(bm[m], f3)
        print("hata:", agin_hatasi)


def net_girdi(x1, w1, x2, w2, bias, b1):
    sonuc = (x1 * w1) + (x2 * w2) + (bias * b1)
    return float(sonuc)

def sigmoid_fonksiyonu(x):
    sonuc = 1 / (1 + math.exp(-x))
    return float(sonuc)

def hata_hesapla(beklenen, cikti):
    hata = beklenen - cikti
    return float(hata)

def yeni_agirlik(eski_agirlik, ogrenme_katsayisi, momentum_katsayisi, agin_ciktisi, agin_hatasi, f1, adm):
    yagirlik = eski_agirlik + agirlik_degisim_miktari(ogrenme_katsayisi, momentum_katsayisi, agin_ciktisi, agin_hatasi,f1, adm)
    return float(yagirlik)

def agirlik_degisim_miktari(ogrenme_katsayisi, momentum_katsayisi, agin_ciktisi, agin_hatasi, f1, adm):
    hata_degeri = agin_ciktisi * (1 - agin_ciktisi) * agin_hatasi
    degisim_miktari = (ogrenme_katsayisi * hata_degeri * f1) + (momentum_katsayisi * adm)
    return float(degisim_miktari)

def girdikatmani_agirlik_degisim_miktari(ogrenme_katsayisi, momentum_katsayisi, agin_ciktisi, agin_hatasi, x, w,eski_agirlik, adm):
    hata_degeri = agin_ciktisi * (1 - agin_ciktisi) * agin_hatasi * w
    degisim_miktari = (ogrenme_katsayisi * hata_degeri * x * eski_agirlik) + (momentum_katsayisi * adm)
    return float(degisim_miktari)

def girdi_katmani_yeni_agirlik(eski_agirlik, ogrenme_katsayisi, momentum_katsayisi, agin_ciktisi, agin_hatasi, x, w, adm):
    yagirlik = eski_agirlik + girdikatmani_agirlik_degisim_miktari(ogrenme_katsayisi, momentum_katsayisi, agin_ciktisi, agin_hatasi, x, w, eski_agirlik, adm)
    return float(yagirlik)

def yeni_bias_degeri(eski_deger, ogrenme_katsayisi, momentum_katsayisi, agin_hatasi, bias_adm):
    yeni_bias = eski_deger + bias_degisim_miktari(ogrenme_katsayisi, momentum_katsayisi, agin_hatasi, bias_adm)
    return float(yeni_bias)

def bias_degisim_miktari(ogrenme_katsayisi, momentum_katsayisi, agin_hatasi, bias_adm):
    bias_degisim_miktari = (ogrenme_katsayisi * agin_hatasi) + (momentum_katsayisi * bias_adm)
    return float(bias_degisim_miktari)

if __name__ == "__main__":
    main()
