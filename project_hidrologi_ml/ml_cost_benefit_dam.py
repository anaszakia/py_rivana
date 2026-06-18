"""
ml_cost_benefit_dam.py — Estimasi Biaya Pembangunan Bendungan/Embung
=====================================================================
Menggantikan MLCostBenefit lama di rivana.py

Sumber data:
  Lapis 1 → Data LPSE PUPR nyata (LPSE_Bendungan_Patched.csv)
             128 paket, 2025-2026, 21 provinsi
  Lapis 2 → AHSP SE DJBK No.68/2024 (fallback)
"""

import os, re, json, math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

LPSE_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LPSE_Bendungan_Patched.csv")

IKK_PROVINSI = {
    "Aceh":1.42,"Sumatera Utara":1.18,"Sumatera Barat":1.15,"Riau":1.28,
    "Kepulauan Riau":1.35,"Jambi":1.20,"Sumatera Selatan":1.14,
    "Bangka Belitung":1.32,"Bengkulu":1.25,"Lampung":1.10,
    "DKI Jakarta":1.12,"Jawa Barat":1.05,"Banten":1.08,
    "Jawa Tengah":1.00,"DI Yogyakarta":1.02,"Jawa Timur":1.03,
    "Bali":1.10,"Nusa Tenggara Barat":1.22,"Nusa Tenggara Timur":1.35,
    "Kalimantan Barat":1.30,"Kalimantan Tengah":1.38,"Kalimantan Selatan":1.25,
    "Kalimantan Timur":1.32,"Kalimantan Utara":1.55,
    "Sulawesi Utara":1.28,"Gorontalo":1.30,"Sulawesi Tengah":1.35,
    "Sulawesi Barat":1.40,"Sulawesi Selatan":1.18,"Sulawesi Tenggara":1.32,
    "Maluku":1.65,"Maluku Utara":1.70,"Papua Barat":2.10,"Papua":2.45,
    "Papua Selatan":2.35,"Papua Tengah":2.40,"Papua Pegunungan":2.50,
    "Papua Barat Daya":2.20,
}
INFLASI = 1.062

AHSP_PER_M3 = {
    "Embung":                       {"min":85000,"median":145000,"max":320000},
    "Kolam Retensi / Long Storage": {"min":120000,"median":195000,"max":450000},
    "Bendungan":                    {"min":65000,"median":110000,"max":280000},
    "Waduk":                        {"min":55000,"median":95000,"max":220000},
    "Groundsill":                   {"min":180000,"median":280000,"max":520000},
    "Check Dam":                    {"min":95000,"median":160000,"max":350000},
}

DURASI = {
    "Embung":                       {"min":6,"max":18,"median":12},
    "Kolam Retensi / Long Storage": {"min":8,"max":24,"median":14},
    "Bendungan":                    {"min":18,"max":48,"median":30},
    "Waduk":                        {"min":36,"max":84,"median":54},
    "Groundsill":                   {"min":6,"max":12,"median":8},
    "Check Dam":                    {"min":4,"max":10,"median":6},
}

KOMPONEN = {
    "Embung": {
        "A. Persiapan & Mobilisasi":0.05,"B. Galian & Pekerjaan Tanah":0.35,
        "C. Timbunan Tubuh Embung":0.28,"D. Pekerjaan Beton & Struktur":0.12,
        "E. Hidromekanikal":0.06,"F. Instrumentasi & Monitoring":0.03,
        "G. Jalan Akses & Penunjang":0.06,"H. Revegetasi & Lingkungan":0.02,
        "I. SMKK (K3 Konstruksi)":0.025,
    },
    "Kolam Retensi / Long Storage": {
        "A. Persiapan & Mobilisasi":0.05,"B. Galian & Pekerjaan Tanah":0.30,
        "C. Timbunan & Pemadatan":0.25,"D. Pekerjaan Beton & Struktur":0.18,
        "E. Hidromekanikal":0.08,"F. Instrumentasi & Monitoring":0.03,
        "G. Jalan Akses & Penunjang":0.06,"H. Revegetasi & Lingkungan":0.02,
        "I. SMKK (K3 Konstruksi)":0.025,
    },
    "Bendungan": {
        "A. Persiapan & Mobilisasi":0.06,"B. Galian & Pekerjaan Tanah":0.28,
        "C. Timbunan Zona Urugan":0.30,"D. Pekerjaan Beton & Struktur":0.14,
        "E. Bangunan Pengelak Sungai":0.05,"F. Hidromekanikal":0.06,
        "G. Instrumentasi & Monitoring":0.04,"H. Jalan Akses & Penunjang":0.05,
        "I. Revegetasi & Lingkungan":0.02,"J. SMKK (K3 Konstruksi)":0.025,
    },
    "Waduk": {
        "A. Persiapan & Mobilisasi":0.06,"B. Galian & Pekerjaan Tanah":0.25,
        "C. Timbunan Zona Urugan":0.30,"D. Pekerjaan Beton & Struktur":0.16,
        "E. Bangunan Pengelak Sungai":0.06,"F. Hidromekanikal":0.07,
        "G. Instrumentasi & Monitoring":0.04,"H. Jalan Akses & Penunjang":0.05,
        "I. Revegetasi & Lingkungan":0.025,"J. SMKK (K3 Konstruksi)":0.025,
    },
    "Groundsill": {
        "A. Persiapan & Mobilisasi":0.05,"B. Galian & Pekerjaan Tanah":0.20,
        "C. Pekerjaan Beton Masif":0.45,"D. Baja & Hidromekanikal":0.10,
        "E. Pekerjaan Penunjang":0.10,"F. SMKK (K3 Konstruksi)":0.025,
    },
}

def get_provinsi_dari_das(das):
    try:
        c = das.geometry.centroid(maxError=100).getInfo()
        lon, lat = c['coordinates'][0], c['coordinates'][1]
    except:
        return "Jawa Tengah"
    if lon<98 and lat>4: return "Aceh"
    elif lon<100 and lat>1: return "Sumatera Utara"
    elif lon<100 and lat>-2: return "Sumatera Barat"
    elif lon<102 and lat>0: return "Riau"
    elif lon<104 and lat>0: return "Kepulauan Riau"
    elif lon<104 and lat>-2: return "Jambi"
    elif lon<106 and lat>-4: return "Sumatera Selatan"
    elif lon<106 and lat>-4: return "Bangka Belitung"
    elif lon<106 and lat>-6: return "Lampung"
    elif lon<107 and lat>-6: return "Banten"
    elif lon<107 and lat>-6.5: return "DKI Jakarta"
    elif lon<109 and lat>-7: return "Jawa Barat"
    elif lon<110.5 and lat>-8: return "DI Yogyakarta" if lat>-8 and lon>109.5 else "Jawa Tengah"
    elif lon<112 and lat>-7.5: return "Jawa Tengah"
    elif lon<115 and lat>-8.8: return "Jawa Timur"
    elif lon<115.5 and lat>-9: return "Bali"
    elif lon<117 and lat>-9: return "Nusa Tenggara Barat"
    elif lon<125 and lat>-11: return "Nusa Tenggara Timur"
    elif lon<109 and lat>-2: return "Kalimantan Barat"
    elif lon<116 and lat>-3: return "Kalimantan Tengah"
    elif lon<117 and lat>1: return "Kalimantan Utara"
    elif lon<118 and lat>-1: return "Kalimantan Timur"
    elif lon<117 and lat>-4: return "Kalimantan Selatan"
    elif lon<122 and lat>1: return "Sulawesi Utara"
    elif lon<122 and lat>-2: return "Gorontalo"
    elif lon<122 and lat>-4: return "Sulawesi Tengah"
    elif lon<120 and lat>-5: return "Sulawesi Barat"
    elif lon<122 and lat>-6: return "Sulawesi Selatan"
    elif lon<124 and lat>-5: return "Sulawesi Tenggara"
    elif lon<128 and lat>-4: return "Maluku Utara"
    elif lon<132 and lat>-6: return "Maluku"
    elif lon<134 and lat>-2: return "Papua Barat"
    else: return "Papua"

def pilih_tipe_bangunan(area_km2, morphology_data):
    relief = morphology_data.get('relief',50)
    slope  = morphology_data.get('slope_mean',5)
    if area_km2 < 5: return "Embung"
    elif area_km2 < 30:
        return "Kolam Retensi / Long Storage" if relief<20 and slope<3 else "Embung"
    elif area_km2 < 50: return "Embung"
    elif area_km2 < 200: return "Bendungan"
    else: return "Waduk"

pilih_tipe_bendungan = pilih_tipe_bangunan  # alias lama

def estimasi_volume_tampungan(area_km2, morphology_data):
    relief = morphology_data.get('relief',50)
    slope  = morphology_data.get('slope_mean',5)
    sar = 8500 if area_km2<5 else 15000 if area_km2<50 else 25000 if area_km2<200 else 35000
    rf = 1.0 + (relief/500)*0.30
    sf = max(0.70, 1.0 - (slope/30)*0.15)
    v  = area_km2 * sar * rf * sf
    h  = max(2.0, relief*0.3)
    return {"v_tampungan_m3":round(v),"a_kolam_m2":round(v/h),
            "h_rata_m":round(h,1),"metode":"SAR Pd T-14-2004-A PUPR"}

class LPSEBenchmark:
    def __init__(self, csv_path=LPSE_CSV_PATH):
        self.df=None; self.df_k=None; self.loaded=False
        if not os.path.exists(csv_path):
            print(f"   ⚠️  LPSE CSV tidak ditemukan: {csv_path}")
            return
        try:
            df = pd.read_csv(csv_path)
            self.df   = df
            self.df_k = df[(df['kategori_paket']=='Pekerjaan Konstruksi')&df['hps_rp'].notna()&(df['hps_rp']>0)].copy()
            self.loaded = True
            print(f"   ✅ Data LPSE: {len(self.df_k)} paket konstruksi dari {len(df)} total")
        except Exception as e:
            print(f"   ⚠️  Gagal load LPSE: {e}")

    def get_benchmark_provinsi(self, tipe, provinsi):
        if not self.loaded: return None
        sub = self.df_k[self.df_k['tipe_bangunan']==tipe]
        sp  = sub[sub['provinsi']==provinsi]['hps_rp'].dropna()
        if len(sp)>=2:
            return {"jumlah_paket":len(sp),"hps_median_rp":float(sp.median()),
                    "hps_min_rp":float(sp.min()),"hps_max_rp":float(sp.max()),
                    "level":"provinsi","sumber":f"LPSE PUPR 2025-2026 — {provinsi}"}
        sa = sub['hps_rp'].dropna()
        if len(sa)>=2:
            return {"jumlah_paket":len(sa),"hps_median_rp":float(sa.median()),
                    "hps_min_rp":float(sa.min()),"hps_max_rp":float(sa.max()),
                    "level":"nasional","sumber":"LPSE PUPR 2025-2026 — semua provinsi"}
        return None

    def get_benchmark_tipe(self, tipe):
        if not self.loaded: return None
        sub = self.df_k[self.df_k['tipe_bangunan']==tipe]['hps_rp'].dropna()
        if len(sub)<2: return None
        return {"jumlah_paket":int(len(sub)),"hps_min_rp":float(sub.min()),
                "hps_q1_rp":float(sub.quantile(0.25)),"hps_median_rp":float(sub.median()),
                "hps_q3_rp":float(sub.quantile(0.75)),"hps_max_rp":float(sub.max()),
                "sumber":"LPSE PUPR 2025-2026"}

    def get_rasio_kontrak(self, tipe=None):
        default = {"median":83.4,"q1":80.4,"q3":88.5,"n":0,"sumber":"Default LPSE 2025-2026"}
        if not self.loaded: return default
        sub = self.df_k[self.df_k['tipe_bangunan']==tipe]['pct_kontrak_vs_hps'].dropna() if tipe else pd.Series()
        if len(sub)<3: sub = self.df_k['pct_kontrak_vs_hps'].dropna()
        if len(sub)==0: return default
        return {"median":float(sub.median()),"q1":float(sub.quantile(0.25)),
                "q3":float(sub.quantile(0.75)),"n":int(len(sub)),"sumber":"LPSE PUPR 2025-2026"}

    def get_paket_serupa(self, tipe, provinsi, n=5):
        if not self.loaded: return []
        sub = self.df_k[self.df_k['tipe_bangunan']==tipe].copy()
        sub['skor'] = sub['provinsi'].apply(lambda p: 2 if p==provinsi else 1)
        sub = sub.sort_values(['skor','tahun_anggaran'],ascending=[False,False])
        cols = [c for c in ['nama_paket','provinsi','tahun_anggaran','pagu_anggaran_rp',
                             'hps_rp','nilai_kontrak_rp','pct_kontrak_vs_hps',
                             'tahapan_status','url_detail'] if c in sub.columns]
        return sub[cols].head(n).to_dict(orient='records')

_lpse_benchmark = None
def get_lpse_benchmark():
    global _lpse_benchmark
    if _lpse_benchmark is None: _lpse_benchmark = LPSEBenchmark()
    return _lpse_benchmark

def buat_jadwal(tipe):
    dur = DURASI.get(tipe,{"min":18,"max":36,"median":24})
    total = dur["median"]
    if tipe in ["Embung","Kolam Retensi / Long Storage"]:
        thn = [("Tahap I: Persiapan & Mobilisasi",0.00,0.08),
               ("Tahap II: Investigasi & Desain Detail",0.05,0.10),
               ("Tahap III: Galian & Persiapan Pondasi",0.12,0.15),
               ("Tahap IV: Konstruksi Tubuh Bangunan",0.22,0.45),
               ("Tahap V: Bangunan Pelimpah & Mekanikal",0.40,0.28),
               ("Tahap VI: Pengisian & Uji Fungsi",0.75,0.15),
               ("Tahap VII: Finishing & Serah Terima",0.88,0.12)]
    else:
        thn = [("Tahap I: Persiapan & Mobilisasi",0.00,0.08),
               ("Tahap II: Investigasi & Desain Detail",0.05,0.10),
               ("Tahap III: Pengelakan Sungai",0.12,0.12),
               ("Tahap IV: Galian & Persiapan Pondasi",0.18,0.15),
               ("Tahap V: Konstruksi Tubuh Bangunan",0.28,0.42),
               ("Tahap VI: Bangunan Pelimpah & Mekanikal",0.42,0.28),
               ("Tahap VII: Pengisian & Impounding",0.78,0.12),
               ("Tahap VIII: Finishing & Serah Terima",0.88,0.12)]
    jadwal_list = []
    for nama, ps, pd2 in thn:
        m = max(1,int(ps*total)+1); s = min(total,m+max(1,int(pd2*total))-1)
        jadwal_list.append({"tahap":nama,"mulai_bulan":m,"selesai_bulan":s,"durasi_bulan":s-m+1})
    return {"total_bulan":total,"total_tahun":round(total/12,1),
            "rentang_bulan":f"{dur['min']}–{dur['max']}","tahapan":jadwal_list}

def hitung_rab(tipe, hps_val, skenario="moderat"):
    fs = {"minimum":0.85,"moderat":1.00,"maksimum":1.20}.get(skenario,1.00)
    nilai_fisik = hps_val * fs
    komp = KOMPONEN.get(tipe, KOMPONEN["Bendungan"])
    rab_detail = {n:{"persentase_pct":round(p*100,2),"jumlah_rp":round(nilai_fisik*p),
                     "referensi":"SE DJBK No.68/2024 + LPSE PUPR 2025-2026"}
                  for n,p in komp.items()}
    overhead = nilai_fisik * 0.15
    ppn      = (nilai_fisik + overhead) * 0.11
    ded      = (nilai_fisik + overhead + ppn) * 0.05
    total_k  = nilai_fisik + overhead + ppn
    total_p  = total_k + ded
    return {"rab_komponen":rab_detail,"subtotal_fisik_rp":round(nilai_fisik),
            "overhead_rp":round(overhead),"ppn_rp":round(ppn),"ded_rp":round(ded),
            "total_konstruksi_rp":round(total_k),"total_proyek_rp":round(total_p),"skenario":skenario}

class MLCostBenefitDam:
    def __init__(self, das, morphology_data):
        self.das       = das
        self.morphology= morphology_data
        self.area_km2  = das.area_km2
        self.provinsi  = get_provinsi_dari_das(das)
        self.ikk       = IKK_PROVINSI.get(self.provinsi, 1.15)
        self.tipe      = pilih_tipe_bangunan(self.area_km2, morphology_data)
        self.vol       = estimasi_volume_tampungan(self.area_km2, morphology_data)
        self.benchmark = get_lpse_benchmark()
        self.hasil     = {}

    def _estimasi_hps(self):
        v   = self.vol["v_tampungan_m3"]
        ikk = self.ikk
        bm  = self.benchmark.get_benchmark_provinsi(self.tipe, self.provinsi)
        if bm:
            f   = ikk / IKK_PROVINSI.get("Jawa Tengah",1.00) * INFLASI
            med = bm["hps_median_rp"] * f
            mn  = bm["hps_min_rp"]    * f
            mx  = bm["hps_max_rp"]    * f
            src = f"LPSE PUPR 2025-2026 — {bm['level']} ({bm['jumlah_paket']} paket)"
        else:
            ahsp = AHSP_PER_M3.get(self.tipe, AHSP_PER_M3["Bendungan"])
            mn   = v * ahsp["min"]    * ikk * INFLASI
            med  = v * ahsp["median"] * ikk * INFLASI
            mx   = v * ahsp["max"]    * ikk * INFLASI
            src  = "AHSP SE DJBK No.68/2024 (LPSE tidak tersedia)"
            bm   = None
        return {"hps_minimum_rp":round(max(mn,4e8)),"hps_moderat_rp":round(max(med,4e8)),
                "hps_maksimum_rp":round(mx),"sumber":src,
                "benchmark_lpse":bm,"benchmark_tipe":self.benchmark.get_benchmark_tipe(self.tipe)}

    def jalankan(self, output_dir="."):
        print("\n"+"="*70)
        print("  ESTIMASI BIAYA PEMBANGUNAN BENDUNGAN/EMBUNG")
        print("="*70)
        print(f"\n  📍 Provinsi DAS  : {self.provinsi}")
        print(f"  📐 Luas DAS      : {self.area_km2:.1f} km²")
        print(f"  🏗️  Tipe Bangunan : {self.tipe}")
        print(f"  💹 IKK Wilayah   : {self.ikk:.3f}")
        print(f"  💧 Vol. Tampungan: {self.vol['v_tampungan_m3']:>14,.0f} m³")

        hps_est = self._estimasi_hps()
        print(f"\n  📊 ESTIMASI HPS:")
        print(f"     Minimum : Rp {hps_est['hps_minimum_rp']:>18,.0f}")
        print(f"     Moderat : Rp {hps_est['hps_moderat_rp']:>18,.0f}")
        print(f"     Maksimum: Rp {hps_est['hps_maksimum_rp']:>18,.0f}")
        print(f"     Sumber  : {hps_est['sumber']}")

        rasio = self.benchmark.get_rasio_kontrak(self.tipe)
        print(f"\n  📉 Rasio Kontrak/HPS (dari data LPSE):")
        print(f"     Median : {rasio['median']:.1f}%  |  Q1: {rasio['q1']:.1f}%  Q3: {rasio['q3']:.1f}%")

        print(f"\n  💰 RAB 3 SKENARIO:")
        rab_s = {}
        for s, k in [("minimum","hps_minimum_rp"),("moderat","hps_moderat_rp"),("maksimum","hps_maksimum_rp")]:
            rab = hitung_rab(self.tipe, hps_est[k], s)
            rab["estimasi_kontrak_rp"] = round(rab["total_proyek_rp"] * rasio["median"]/100)
            rab_s[s] = rab
            print(f"     {s.capitalize():10s}: Rp {rab['total_proyek_rp']:>20,.0f}  "
                  f"(kontrak ~Rp {rab['estimasi_kontrak_rp']:>18,.0f})")

        jadwal = buat_jadwal(self.tipe)
        print(f"\n  📅 JADWAL: {jadwal['total_bulan']} bulan ({jadwal['total_tahun']} tahun) | Rentang: {jadwal['rentang_bulan']} bln")
        for t in jadwal["tahapan"]:
            print(f"     Bln {t['mulai_bulan']:>2}–{t['selesai_bulan']:>2}  {t['tahap']}")

        paket_serupa = self.benchmark.get_paket_serupa(self.tipe, self.provinsi, n=5)
        if paket_serupa:
            print(f"\n  🔍 PAKET LPSE SERUPA ({len(paket_serupa)} referensi):")
            for p in paket_serupa:
                hps_p = p.get('hps_rp',0) or 0
                print(f"     • {str(p.get('nama_paket',''))[:55]}")
                print(f"       [{p.get('tahun_anggaran','')}] {p.get('provinsi','')} — HPS: Rp {hps_p:,.0f}")

        self.hasil = {
            "metadata": {
                "das_name":self.das.name,"das_area_km2":self.area_km2,
                "provinsi":self.provinsi,"ikk_wilayah":self.ikk,
                "tipe_bangunan":self.tipe,"tanggal_estimasi":datetime.now().strftime("%Y-%m-%d"),
                "referensi":["Data LPSE PUPR 2025-2026 (LPSE_Bendungan_Patched.csv — 128 paket, 21 provinsi)",
                             "SE Ditjen Bina Konstruksi No.68/2024 — AHSP SDA","Permen PUPR No.1/PRT/M/2022",
                             "BPS IKK 2024","Pd T-14-2004-A PUPR"],
                "disclaimer":"Estimasi ORDER OF MAGNITUDE (±30%) untuk studi kelayakan awal.",
            },
            "dimensi_tampungan":self.vol,"estimasi_hps":hps_est,
            "rasio_kontrak_hps":rasio,
            "skenario":{s:{"total_proyek_rp":rab_s[s]["total_proyek_rp"],
                            "estimasi_kontrak_rp":rab_s[s]["estimasi_kontrak_rp"],
                            "breakdown":{"fisik_rp":rab_s[s]["subtotal_fisik_rp"],
                                         "overhead_rp":rab_s[s]["overhead_rp"],
                                         "ppn_rp":rab_s[s]["ppn_rp"],
                                         "ded_rp":rab_s[s]["ded_rp"]},
                            "komponen_biaya":rab_s[s]["rab_komponen"]}
                       for s in ["minimum","moderat","maksimum"]},
            "jadwal":jadwal,"paket_lpse_serupa":paket_serupa,
        }

        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir,"RIVANA_Dam_Cost_Estimate.json")
        with open(out,"w",encoding="utf-8") as f:
            json.dump(self.hasil,f,ensure_ascii=False,indent=2,default=str)
        print(f"\n  ✅ Hasil saved: {out}")
        print("="*70)
        return self.hasil

    def train(self, df_hasil): return df_hasil

    def analyze(self, df_hasil):
        if not self.hasil: return df_hasil
        mod = self.hasil.get("skenario",{}).get("moderat",{})
        df_hasil["dam_tipe"]               = self.tipe
        df_hasil["dam_provinsi"]           = self.provinsi
        df_hasil["dam_total_proyek_rp"]    = mod.get("total_proyek_rp",0)
        df_hasil["dam_estimasi_kontrak_rp"]= mod.get("estimasi_kontrak_rp",0)
        df_hasil["dam_v_tampungan_m3"]     = self.vol["v_tampungan_m3"]
        df_hasil["dam_durasi_bulan"]       = self.hasil.get("jadwal",{}).get("total_bulan",0)
        df_hasil["dam_sumber_data"]        = "LPSE PUPR 2025-2026" if self.benchmark.loaded else "AHSP SE DJBK 68/2024"
        return df_hasil