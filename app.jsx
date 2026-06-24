import React, { useState, useEffect, useMemo } from 'react';
import { 
  Activity, 
  Heart, 
  User, 
  Settings, 
  AlertCircle, 
  BarChart2, 
  ShieldAlert, 
  CheckCircle2, 
  RefreshCw, 
  HelpCircle, 
  ChevronRight, 
  Table, 
  Sliders, 
  TrendingUp, 
  Info,
  ChevronLeft,
  BookOpen
} from 'lucide-react';

// ==============================================================================
// 1. GENERATOR DATASET SINTETIS (Sesuai dengan logika Python np.random.seed(42))
// ==============================================================================
// Menggunakan Linear Congruential Generator (LCG) sederhana agar menghasilkan angka acak yang konsisten
const createPseudoRandom = (seed) => {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
};

const generateDataset = () => {
  const random = createPseudoRandom(42);
  const nSamples = 60;
  const data = [];

  // Rentang parameter medis
  for (let i = 0; i < nSamples; i++) {
    const umur = Math.floor(random() * (74 - 30 + 1)) + 30;               // 30 - 74 tahun
    const tekananDarah = Math.floor(random() * (179 - 90 + 1)) + 90;       // 90 - 179 mmHg
    const kolesterol = Math.floor(random() * (319 - 150 + 1)) + 150;       // 150 - 319 mg/dL
    const detakJantung = Math.floor(random() * (109 - 60 + 1)) + 60;       // 60 - 109 bpm

    // Logika perhitungan risiko medis buatan (sama dengan versi Python)
    const skorRisiko = (umur * 0.3) + (tekananDarah * 0.4) + (kolesterol * 0.2) + (detakJantung * 0.1);
    data.push({ id: i + 1, umur, tekananDarah, kolesterol, detakJantung, skorRisiko });
  }

  // Tentukan median skor untuk klasifikasi biner 'Tinggi' / 'Rendah'
  const sortedScores = [...data].map(d => d.skorRisiko).sort((a, b) => a - b);
  const median = sortedScores[Math.floor(sortedScores.length / 2)];

  return data.map(d => ({
    id: d.id,
    umur: d.umur,
    tekananDarah: d.tekananDarah,
    kolesterol: d.kolesterol,
    detakJantung: d.detakJantung,
    risiko: d.skorRisiko >= median ? 'Tinggi' : 'Rendah'
  }));
};

// ==============================================================================
// 2. ALGORITMA UTAMA MACHINE LEARNING (KNN & STANDARDIZATION)
// ==============================================================================

// Menghitung Mean dan Standard Deviation dari Data Latih (X_train)
const calculateStats = (X) => {
  const numFeatures = X[0].length;
  const stats = [];
  for (let j = 0; j < numFeatures; j++) {
    const values = X.map(row => row[j]);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance) || 1; // Hindari pembagian dengan nol
    stats.push({ mean, std });
  }
  return stats;
};

// Melakukan transformasi skala (Z-score normalization)
const standardizeFeatures = (X, stats) => {
  return X.map(row => row.map((val, j) => (val - stats[j].mean) / stats[j].std));
};

// Perhitungan Jarak Euclidean
const calculateEuclideanDistance = (point1, point2) => {
  return Math.sqrt(point1.reduce((sum, val, i) => sum + Math.pow(val - point2[i], 2), 0));
};

// Fungsi Prediksi KNN untuk Satu Titik Uji
const knnPredictSingle = (X_train, y_train, testPoint, k) => {
  // Hitung jarak ke seluruh data latih
  const distances = X_train.map((trainPoint, idx) => ({
    distance: calculateEuclideanDistance(trainPoint, testPoint),
    label: y_train[idx]
  }));

  // Urutkan jarak terdekat (ascending)
  distances.sort((a, b) => a.distance - b.distance);

  // Ambil K tetangga terdekat
  const neighbors = distances.slice(0, k);

  // Lakukan voting suara terbanyak
  const votes = { Tinggi: 0, Rendah: 0 };
  neighbors.forEach(n => {
    votes[n.label] = (votes[n.label] || 0) + 1;
  });

  const prediction = votes.Tinggi >= votes.Rendah ? 'Tinggi' : 'Rendah';
  const confidence = (votes[prediction] / k) * 100;

  return { prediction, confidence, neighbors };
};

export default function App() {
  // State Utama Aplikasi
  const [dataset, setDataset] = useState([]);
  const [kValue, setKValue] = useState(3);
  const [trainRatio, setTrainRatio] = useState(80); // Persentase data latih (80%)
  const [useScaling, setUseScaling] = useState(true); // Toggle standardisasi scaler
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard', 'dataset', 'predict', 'theory'
  
  // State Form Input Pasien Baru
  const [inputPasien, setInputPasien] = useState({
    umur: 55,
    tekananDarah: 140,
    kolesterol: 240,
    detakJantung: 85
  });
  
  // State Hasil Prediksi Pasien Baru
  const [hasilPrediksiPasien, setHasilPrediksiPasien] = useState(null);

  // State Halaman Tabel Dataset
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  // Inisialisasi Dataset saat pertama kali dimuat
  useEffect(() => {
    setDataset(generateDataset());
  }, []);

  // ==============================================================================
  // 3. PIPELINE PROSES MACHINE LEARNING
  // ==============================================================================
  const mlResults = useMemo(() => {
    if (dataset.length === 0) return null;

    // A. Memisahkan Fitur (X) dan Target (y)
    // Fitur: [Umur, TekananDarah, Kolesterol, DetakJantung]
    const X = dataset.map(d => [d.umur, d.tekananDarah, d.kolesterol, d.detakJantung]);
    const y = dataset.map(d => d.risiko);

    // B. Pembagian Data Training & Testing (Stratified Train-Test Split manual)
    // Mengelompokkan indeks berdasarkan kelas untuk menjaga proporsi
    const tinggiIndices = [];
    const rendahIndices = [];
    y.forEach((val, idx) => {
      if (val === 'Tinggi') tinggiIndices.push(idx);
      else rendahIndices.push(idx);
    });

    // Menghitung batas jumlah data latih
    const trainCountTinggi = Math.round(tinggiIndices.length * (trainRatio / 100));
    const trainCountRendah = Math.round(rendahIndices.length * (trainRatio / 100));

    // Membagi indeks secara deterministik (reproducible)
    const trainIndices = [
      ...tinggiIndices.slice(0, trainCountTinggi),
      ...rendahIndices.slice(0, trainCountRendah)
    ];
    const testIndices = [
      ...tinggiIndices.slice(trainCountTinggi),
      ...rendahIndices.slice(trainCountRendah)
    ];

    // Membuat array X_train, y_train, X_test, y_test
    const X_train = trainIndices.map(idx => X[idx]);
    const y_train = trainIndices.map(idx => y[idx]);
    const X_test = testIndices.map(idx => X[idx]);
    const y_test = testIndices.map(idx => y[idx]);

    // C. Normalisasi Fitur (Z-score Normalization)
    let X_train_processed = X_train;
    let X_test_processed = X_test;
    let scalerStats = null;

    if (useScaling) {
      scalerStats = calculateStats(X_train);
      X_train_processed = standardizeFeatures(X_train, scalerStats);
      X_test_processed = standardizeFeatures(X_test, scalerStats);
    }

    // D. Evaluasi Model pada Data Test
    const predictions = X_test_processed.map(testPoint => {
      return knnPredictSingle(X_train_processed, y_train, testPoint, kValue).prediction;
    });

    // E. Perhitungan Metrik Akurasi
    let correct = 0;
    predictions.forEach((pred, idx) => {
      if (pred === y_test[idx]) correct++;
    });
    const accuracy = correct / y_test.length;

    // F. Perhitungan Confusion Matrix
    // Aktual \ Prediksi: Rendah (0), Tinggi (1)
    let tn = 0; // Aktual Rendah, Prediksi Rendah
    let fp = 0; // Aktual Rendah, Prediksi Tinggi
    let fn = 0; // Aktual Tinggi, Prediksi Rendah
    let tp = 0; // Aktual Tinggi, Prediksi Tinggi

    y_test.forEach((actual, idx) => {
      const pred = predictions[idx];
      if (actual === 'Rendah' && pred === 'Rendah') tn++;
      else if (actual === 'Rendah' && pred === 'Tinggi') fp++;
      else if (actual === 'Tinggi' && pred === 'Rendah') fn++;
      else if (actual === 'Tinggi' && pred === 'Tinggi') tp++;
    });

    // G. Perhitungan Metrik Tambahan (Precision, Recall, F1)
    const precisionTinggi = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recallTinggi = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Tinggi = precisionTinggi + recallTinggi > 0 ? 2 * (precisionTinggi * recallTinggi) / (precisionTinggi + recallTinggi) : 0;

    const precisionRendah = tn + fn > 0 ? tn / (tn + fn) : 0;
    const recallRendah = tn + fp > 0 ? tn / (tn + fp) : 0;
    const f1Rendah = precisionRendah + recallRendah > 0 ? 2 * (precisionRendah * recallRendah) / (precisionRendah + recallRendah) : 0;

    // H. Menghitung kurva performa akurasi vs K untuk visualisasi line chart (K = 1 sampai 15)
    const kCurveData = Array.from({ length: 15 }, (_, i) => {
      const k = i + 1;
      const tempPredictions = X_test_processed.map(testPoint => {
        return knnPredictSingle(X_train_processed, y_train, testPoint, k).prediction;
      });
      let tempCorrect = 0;
      tempPredictions.forEach((pred, idx) => {
        if (pred === y_test[idx]) tempCorrect++;
      });
      return { k, accuracy: tempCorrect / y_test.length };
    });

    return {
      X_train_processed,
      y_train,
      scalerStats,
      accuracy,
      confusionMatrix: { tn, fp, fn, tp },
      metrics: {
        tinggi: { precision: precisionTinggi, recall: recallTinggi, f1: f1Tinggi, count: tp + fn },
        rendah: { precision: precisionRendah, recall: recallRendah, f1: f1Rendah, count: tn + fp }
      },
      kCurveData,
      testCount: X_test.length,
      trainCount: X_train.length
    };
  }, [dataset, kValue, trainRatio, useScaling]);

  // Trigger Prediksi Pasien Baru secara otomatis saat parameter input atau model berubah
  useEffect(() => {
    if (mlResults) {
      handlePrediksi();
    }
  }, [inputPasien, mlResults]);

  const handlePrediksi = () => {
    if (!mlResults) return;

    const queryPoint = [
      inputPasien.umur,
      inputPasien.tekananDarah,
      inputPasien.kolesterol,
      inputPasien.detakJantung
    ];

    let processedPoint = queryPoint;
    if (useScaling && mlResults.scalerStats) {
      processedPoint = queryPoint.map((val, idx) => {
        const stats = mlResults.scalerStats[idx];
        return (val - stats.mean) / stats.std;
      });
    }

    const res = knnPredictSingle(
      mlResults.X_train_processed,
      mlResults.y_train,
      processedPoint,
      kValue
    );

    setHasilPrediksiPasien(res);
  };

  // Hitung jumlah kelas untuk visualisasi bar chart distribusi target
  const classDistribution = useMemo(() => {
    const counts = { Tinggi: 0, Rendah: 0 };
    dataset.forEach(d => {
      counts[d.risiko]++;
    });
    return counts;
  }, [dataset]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
      
      {/* ==============================================================================
          HEADER UTAMA
          ============================================================================== */}
      <header className="bg-gradient-to-r from-blue-700 via-indigo-700 to-slate-900 text-white shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="bg-rose-500 p-2 rounded-xl text-white shadow-inner animate-pulse">
              <Activity className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl md:text-2xl font-bold tracking-tight flex items-center gap-2">
                CardioKNN <span className="text-xs bg-indigo-500/50 px-2 py-0.5 rounded-full uppercase border border-indigo-400">Dashboard Edukasi</span>
              </h1>
              <p className="text-xs text-indigo-100 mt-0.5">Sistem Klasifikasi Risiko Penyakit Jantung dengan Algoritma K-Nearest Neighbor</p>
            </div>
          </div>
          
          {/* Menu Navigasi */}
          <nav className="flex space-x-1 bg-indigo-900/30 p-1 rounded-lg border border-indigo-500/20">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === 'dashboard' ? 'bg-indigo-600 text-white shadow' : 'text-indigo-200 hover:bg-indigo-800/20 hover:text-white'
              }`}
            >
              <TrendingUp className="h-3.5 w-3.5 inline mr-1" />
              Eksperimen Model
            </button>
            <button
              onClick={() => setActiveTab('dataset')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === 'dataset' ? 'bg-indigo-600 text-white shadow' : 'text-indigo-200 hover:bg-indigo-800/20 hover:text-white'
              }`}
            >
              <Table className="h-3.5 w-3.5 inline mr-1" />
              Eksplorasi Data
            </button>
            <button
              onClick={() => setActiveTab('predict')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === 'predict' ? 'bg-indigo-600 text-white shadow' : 'text-indigo-200 hover:bg-indigo-800/20 hover:text-white'
              }`}
            >
              <Heart className="h-3.5 w-3.5 inline mr-1" />
              Prediksi Pasien
            </button>
            <button
              onClick={() => setActiveTab('theory')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === 'theory' ? 'bg-indigo-600 text-white shadow' : 'text-indigo-200 hover:bg-indigo-800/20 hover:text-white'
              }`}
            >
              <BookOpen className="h-3.5 w-3.5 inline mr-1" />
              Teori Dasar
            </button>
          </nav>
        </div>
      </header>

      {/* ==============================================================================
          MAIN CONTAINER
          ============================================================================== */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:py-6">
        
        {/* Banner Penjelasan Ringkas */}
        <div className="bg-indigo-50 border-l-4 border-indigo-600 p-4 rounded-r-xl shadow-sm mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start space-x-3">
            <Info className="h-5 w-5 text-indigo-600 shrink-0 mt-0.5" />
            <div>
              <h2 className="text-sm font-semibold text-indigo-900">Media Interaktif Pembelajaran Algoritma KNN</h2>
              <p className="text-xs text-indigo-700 mt-1">
                Gunakan panel konfigurasi di bawah untuk mengontrol parameter hyperparameter model ML. Amati pergerakan nilai akurasi, confusion matrix, dan tingkat sensitivitas klasifikasi secara instan saat Anda mengubah parameter.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-xs font-semibold bg-indigo-200/50 text-indigo-800 px-2.5 py-1 rounded-full border border-indigo-200">
              {dataset.length} Data Pasien (CSV)
            </span>
          </div>
        </div>

        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* ==============================================================================
                KOLOM KIRI: PANEL PENGATURAN PARAMETER MODEL (Span 4)
                ============================================================================== */}
            <div className="lg:col-span-4 space-y-6">
              <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200/80">
                <div className="flex items-center space-x-2 pb-4 mb-4 border-b border-slate-100">
                  <Settings className="h-5 w-5 text-indigo-600" />
                  <h3 className="font-bold text-slate-800 text-base">Konfigurasi Model (Hyperparameter)</h3>
                </div>

                {/* Slider Nilai K (Tetangga Terdekat) */}
                <div className="space-y-2 mb-6">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-semibold text-slate-600 flex items-center gap-1.5">
                      Nilai K (Tetangga Terdekat)
                      <HelpCircle className="h-3.5 w-3.5 text-slate-400 cursor-help" title="Jumlah tetangga terdekat yang digunakan untuk proses voting suara mayoritas." />
                    </label>
                    <span className="bg-indigo-100 text-indigo-800 px-2.5 py-0.5 rounded-full text-xs font-extrabold border border-indigo-200">
                      k = {kValue}
                    </span>
                  </div>
                  <input 
                    type="range" 
                    min="1" 
                    max="15" 
                    step="1"
                    value={kValue}
                    onChange={(e) => setKValue(parseInt(e.target.value))}
                    className="w-full accent-indigo-600 cursor-pointer h-2 bg-slate-100 rounded-lg border-none"
                  />
                  <div className="flex justify-between text-[10px] text-slate-400 font-medium">
                    <span>1 (Sangat Sensitif)</span>
                    <span>15 (Sangat Stabil)</span>
                  </div>
                  {kValue % 2 === 0 && (
                    <div className="bg-amber-50 border border-amber-200 p-2.5 rounded-lg flex items-start space-x-2 mt-2">
                      <AlertCircle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
                      <p className="text-[10px] text-amber-800 leading-relaxed">
                        <strong>Catatan Nilai K Genap:</strong> Menggunakan nilai K genap ({kValue}) berisiko menghasilkan suara seri (*draw*) saat voting. Nilai ganjil lebih direkomendasikan untuk klasifikasi biner.
                      </p>
                    </div>
                  )}
                </div>

                {/* Slider Rasio Pembagian Data (Train-Test Split) */}
                <div className="space-y-2 mb-6">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-semibold text-slate-600">
                      Rasio Data Latih (Training)
                    </label>
                    <span className="bg-blue-50 text-blue-800 px-2.5 py-0.5 rounded-full text-xs font-extrabold border border-blue-200">
                      {trainRatio} : {100 - trainRatio}
                    </span>
                  </div>
                  <input 
                    type="range" 
                    min="50" 
                    max="90" 
                    step="5"
                    value={trainRatio}
                    onChange={(e) => setTrainRatio(parseInt(e.target.value))}
                    className="w-full accent-blue-600 cursor-pointer h-2 bg-slate-100 rounded-lg border-none"
                  />
                  <div className="flex justify-between text-[10px] text-slate-400 font-medium">
                    <span>Latih 50% / Uji 50%</span>
                    <span>Latih 90% / Uji 10%</span>
                  </div>
                  <div className="flex items-center justify-between mt-2 bg-slate-50 p-2 rounded-lg text-[11px] text-slate-600 border border-slate-100">
                    <span>Data Latih: <strong>{mlResults?.trainCount}</strong></span>
                    <span>Data Uji: <strong>{mlResults?.testCount}</strong></span>
                  </div>
                </div>

                {/* Toggle Standardisasi Data (StandardScaler) */}
                <div className="pt-4 border-t border-slate-100">
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-xs font-semibold text-slate-700 block">
                        Normalisasi Data (StandardScaler)
                      </label>
                      <span className="text-[10px] text-slate-400 block mt-0.5">Konversi semua fitur ke skala standar (Z-score)</span>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input 
                        type="checkbox" 
                        checked={useScaling}
                        onChange={() => setUseScaling(!useScaling)}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
                    </label>
                  </div>

                  {!useScaling && (
                    <div className="bg-rose-50 border border-rose-200 p-2.5 rounded-lg flex items-start space-x-2 mt-3">
                      <ShieldAlert className="h-4 w-4 text-rose-500 shrink-0 mt-0.5" />
                      <p className="text-[10px] text-rose-800 leading-relaxed">
                        <strong>Perhatian:</strong> Tanpa standardisasi, fitur dengan nilai nominal tinggi seperti Kolesterol (150-320 mg/dL) akan mendominasi metrik jarak dibanding Umur (30-74 thn), sehingga akurasi prediksi model biasanya merosot tajam.
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Panel Ringkasan Metrik Akurasi */}
              <div className="bg-gradient-to-br from-indigo-800 to-slate-900 text-white p-5 rounded-2xl shadow-sm border border-indigo-950">
                <h4 className="text-xs uppercase tracking-wider text-indigo-200 font-bold mb-1">Akurasi Pengujian Model</h4>
                <div className="flex items-baseline space-x-2">
                  <span className="text-4xl font-extrabold tracking-tight">
                    {mlResults ? (mlResults.accuracy * 100).toFixed(1) : '0.0'}%
                  </span>
                  <span className="text-xs text-indigo-300">Akurasi Klasifikasi</span>
                </div>
                
                {/* Progress bar akurasi */}
                <div className="w-full bg-indigo-950 rounded-full h-2.5 mt-3 overflow-hidden border border-indigo-700/30">
                  <div 
                    className="bg-emerald-500 h-2.5 rounded-full transition-all duration-500" 
                    style={{ width: `${mlResults ? mlResults.accuracy * 100 : 0}%` }}
                  ></div>
                </div>

                <div className="grid grid-cols-2 gap-2 mt-4 pt-4 border-t border-indigo-800/50 text-xs text-indigo-100">
                  <div className="bg-indigo-950/40 p-2 rounded-lg">
                    <span className="text-[10px] text-indigo-300 block">Kebenaran Prediksi</span>
                    <strong className="text-sm">
                      {mlResults ? Math.round(mlResults.accuracy * mlResults.testCount) : 0} / {mlResults?.testCount} Data Uji
                    </strong>
                  </div>
                  <div className="bg-indigo-950/40 p-2 rounded-lg">
                    <span className="text-[10px] text-indigo-300 block">Skala Fitur</span>
                    <strong className="text-sm text-emerald-400">
                      {useScaling ? "Aktif (Z-score)" : "Tidak Aktif"}
                    </strong>
                  </div>
                </div>
              </div>
            </div>

            {/* ==============================================================================
                KOLOM KANAN: HASIL EVALUASI & VISUALISASI MODEL (Span 8)
                ============================================================================== */}
            <div className="lg:col-span-8 space-y-6">
              
              {/* Grid Atas: Evaluasi Metrik & Confusion Matrix */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Klasifikasi Detail (Classification Report) */}
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200/80 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
                      <h4 className="font-bold text-slate-800 text-sm flex items-center gap-1.5">
                        <BarChart2 className="h-4 w-4 text-indigo-600" />
                        Laporan Klasifikasi
                      </h4>
                      <span className="text-[10px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded-md font-medium">Metrik Detil</span>
                    </div>

                    <div className="space-y-4">
                      {/* Kelas Risiko Tinggi */}
                      <div>
                        <div className="flex justify-between items-center text-xs mb-1">
                          <span className="font-bold text-slate-700 flex items-center gap-1.5">
                            <span className="h-2 w-2 rounded-full bg-rose-500"></span>
                            Risiko Tinggi
                          </span>
                          <span className="text-slate-500 text-[10px]">Data: {mlResults?.metrics.tinggi.count} pasien</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-center">
                          <div className="bg-slate-50 p-1.5 rounded border border-slate-100">
                            <span className="text-[9px] text-slate-400 block uppercase">Precision</span>
                            <span className="font-semibold text-xs text-slate-700">{(mlResults?.metrics.tinggi.precision * 100).toFixed(0)}%</span>
                          </div>
                          <div className="bg-slate-50 p-1.5 rounded border border-slate-100">
                            <span className="text-[9px] text-slate-400 block uppercase">Recall</span>
                            <span className="font-semibold text-xs text-slate-700">{(mlResults?.metrics.tinggi.recall * 100).toFixed(0)}%</span>
                          </div>
                          <div className="bg-slate-50 p-1.5 rounded border border-slate-100">
                            <span className="text-[9px] text-slate-400 block uppercase">F1-Score</span>
                            <span className="font-semibold text-xs text-slate-700">{(mlResults?.metrics.tinggi.f1 * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>

                      {/* Kelas Risiko Rendah */}
                      <div>
                        <div className="flex justify-between items-center text-xs mb-1">
                          <span className="font-bold text-slate-700 flex items-center gap-1.5">
                            <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
                            Risiko Rendah
                          </span>
                          <span className="text-slate-500 text-[10px]">Data: {mlResults?.metrics.rendah.count} pasien</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-center">
                          <div className="bg-slate-50 p-1.5 rounded border border-slate-100">
                            <span className="text-[9px] text-slate-400 block uppercase">Precision</span>
                            <span className="font-semibold text-xs text-slate-700">{(mlResults?.metrics.rendah.precision * 100).toFixed(0)}%</span>
                          </div>
                          <div className="bg-slate-50 p-1.5 rounded border border-slate-100">
                            <span className="text-[9px] text-slate-400 block uppercase">Recall</span>
                            <span className="font-semibold text-xs text-slate-700">{(mlResults?.metrics.rendah.recall * 100).toFixed(0)}%</span>
                          </div>
                          <div className="bg-slate-50 p-1.5 rounded border border-slate-100">
                            <span className="text-[9px] text-slate-400 block uppercase">F1-Score</span>
                            <span className="font-semibold text-xs text-slate-700">{(mlResults?.metrics.rendah.f1 * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-400 mt-4 leading-normal italic">
                    * Precision mengukur ketepatan deteksi risiko, Recall mengukur jangkauan penemuan kasus aktual yang sakit.
                  </p>
                </div>

                {/* Heatmap Confusion Matrix */}
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200/80">
                  <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
                    <h4 className="font-bold text-slate-800 text-sm flex items-center gap-1.5">
                      <Sliders className="h-4 w-4 text-indigo-600" />
                      Heatmap Confusion Matrix
                    </h4>
                    <span className="text-[10px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded-md font-medium">Prediksi vs Kenyataan</span>
                  </div>

                  <div className="flex flex-col items-center">
                    <div className="grid grid-cols-3 gap-2 w-full max-w-[280px]">
                      {/* Baris Pertama Label Kolom */}
                      <div></div>
                      <div className="text-center text-[10px] font-bold text-slate-400 uppercase">Pred Rendah</div>
                      <div className="text-center text-[10px] font-bold text-slate-400 uppercase">Pred Tinggi</div>

                      {/* Baris Aktual Rendah */}
                      <div className="flex items-center justify-end pr-2 text-[10px] font-bold text-slate-400 uppercase text-right leading-none">
                        Aktual Rendah
                      </div>
                      <div className="aspect-square bg-emerald-100 rounded-xl border border-emerald-200 flex flex-col items-center justify-center text-center p-1">
                        <span className="text-lg font-black text-emerald-800">{mlResults?.confusionMatrix.tn}</span>
                        <span className="text-[8px] text-emerald-600 uppercase font-semibold">True Neg</span>
                      </div>
                      <div className={`aspect-square rounded-xl border flex flex-col items-center justify-center text-center p-1 ${
                        mlResults?.confusionMatrix.fp > 0 ? 'bg-rose-50 border-rose-200 text-rose-800' : 'bg-slate-50 border-slate-200 text-slate-400'
                      }`}>
                        <span className="text-lg font-black">{mlResults?.confusionMatrix.fp}</span>
                        <span className="text-[8px] uppercase font-semibold">False Pos</span>
                      </div>

                      {/* Baris Aktual Tinggi */}
                      <div className="flex items-center justify-end pr-2 text-[10px] font-bold text-slate-400 uppercase text-right leading-none">
                        Aktual Tinggi
                      </div>
                      <div className={`aspect-square rounded-xl border flex flex-col items-center justify-center text-center p-1 ${
                        mlResults?.confusionMatrix.fn > 0 ? 'bg-rose-50 border-rose-200 text-rose-800' : 'bg-slate-50 border-slate-200 text-slate-400'
                      }`}>
                        <span className="text-lg font-black">{mlResults?.confusionMatrix.fn}</span>
                        <span className="text-[8px] uppercase font-semibold">False Neg</span>
                      </div>
                      <div className="aspect-square bg-emerald-100 rounded-xl border border-emerald-200 flex flex-col items-center justify-center text-center p-1">
                        <span className="text-lg font-black text-emerald-800">{mlResults?.confusionMatrix.tp}</span>
                        <span className="text-[8px] text-emerald-600 uppercase font-semibold">True Pos</span>
                      </div>
                    </div>
                  </div>
                </div>

              </div>

              {/* Kurva Grafik Akurasi vs Nilai K (Line Chart Menggunakan Pure SVG) */}
              <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200/80">
                <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
                  <h4 className="font-bold text-slate-800 text-sm flex items-center gap-1.5">
                    <TrendingUp className="h-4 w-4 text-indigo-600" />
                    Grafik Akurasi Model Terhadap Berbagai Nilai K (1 - 15)
                  </h4>
                  <span className="text-[10px] bg-indigo-50 text-indigo-700 px-2.5 py-1 rounded-full font-bold border border-indigo-100">
                    K Terbaik = K-{mlResults?.kCurveData.reduce((best, cur) => cur.accuracy > best.accuracy ? cur : best, { k: 3, accuracy: 0 }).k}
                  </span>
                </div>

                <div className="w-full h-56 pt-2">
                  <svg viewBox="0 0 600 220" className="w-full h-full overflow-visible">
                    {/* Grid Garis Horizontal */}
                    {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((yVal, i) => {
                      const yPos = 200 - yVal * 180;
                      return (
                        <g key={i}>
                          <line x1="40" y1={yPos} x2="580" y2={yPos} stroke="#f1f5f9" strokeWidth="1.5" />
                          <text x="15" y={yPos + 4} fill="#94a3b8" className="text-[10px] font-semibold text-right">{yVal * 100}%</text>
                        </g>
                      );
                    })}

                    {/* Plot Garis Akurasi */}
                    {mlResults && (
                      <>
                        {/* Membuat polyline koordinat (x, y) */}
                        {(() => {
                          const points = mlResults.kCurveData.map(d => {
                            const x = 40 + (d.k - 1) * (540 / 14);
                            const y = 200 - d.accuracy * 180;
                            return `${x},${y}`;
                          }).join(' ');

                          return (
                            <polyline
                              fill="none"
                              stroke="#6366f1"
                              strokeWidth="3"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              points={points}
                            />
                          );
                        })()}

                        {/* Menggambar Marker Titik Bundar & Highlight Nilai K Aktif */}
                        {mlResults.kCurveData.map(d => {
                          const x = 40 + (d.k - 1) * (540 / 14);
                          const y = 200 - d.accuracy * 180;
                          const isActive = d.k === kValue;

                          return (
                            <g key={d.k} className="group">
                              {/* Lingkaran Titik Koordinat */}
                              <circle
                                cx={x}
                                cy={y}
                                r={isActive ? 6 : 4}
                                fill={isActive ? '#dc2626' : '#6366f1'}
                                stroke="#ffffff"
                                strokeWidth="1.5"
                                className="cursor-pointer transition-all duration-300"
                              />
                              
                              {/* Indikator vertikal putus-putus untuk K saat ini */}
                              {isActive && (
                                <line 
                                  x1={x} 
                                  y1={y} 
                                  x2={x} 
                                  y2="200" 
                                  stroke="#dc2626" 
                                  strokeWidth="1.5" 
                                  strokeDasharray="3,3" 
                                />
                              )}

                              {/* Label Angka Akurasi di Atas Titik */}
                              <text 
                                x={x} 
                                y={y - 8} 
                                fill={isActive ? '#dc2626' : '#475569'} 
                                className={`text-[9px] text-center font-bold font-mono`} 
                                textAnchor="middle"
                              >
                                {(d.accuracy * 100).toFixed(0)}%
                              </text>

                              {/* Label Sumbu X (Nilai K) */}
                              <text 
                                x={x} 
                                y="215" 
                                fill={isActive ? '#dc2626' : '#94a3b8'} 
                                className={`text-[10px] font-bold ${isActive ? 'font-black' : ''}`} 
                                textAnchor="middle"
                              >
                                K-{d.k}
                              </text>
                            </g>
                          );
                        })}
                      </>
                    )}
                  </svg>
                </div>
                <div className="flex justify-center items-center gap-4 mt-2 text-[10px] text-slate-500 font-medium">
                  <div className="flex items-center gap-1">
                    <span className="h-2 w-5 bg-indigo-500 rounded-full inline-block"></span>
                    <span>Tingkat Akurasi</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="h-2.5 w-2.5 bg-rose-600 rounded-full inline-block"></span>
                    <span>K Pilihan Anda saat ini ({kValue})</span>
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* ==============================================================================
            TAB: TABEL DATASET (Eksplorasi Data)
            ============================================================================== */}
        {activeTab === 'dataset' && (
          <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200/80 space-y-6">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-4 border-b border-slate-100">
              <div>
                <h3 className="font-bold text-slate-800 text-lg flex items-center gap-2">
                  <Table className="h-5 w-5 text-indigo-600" />
                  Eksplorasi Dataset Pasien Jantung (`heart_risk.csv`)
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  Menampilkan tabel dataset sintetis medis yang tersimpan di dalam berkas CSV. Anda dapat memantau komposisi fitur dan tingkat keparahan risiko jantung.
                </p>
              </div>

              {/* Bar Distribusi Data */}
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-100 flex items-center gap-6 shrink-0">
                <div className="text-xs">
                  <span className="text-slate-400 block font-medium">Distribusi Risiko</span>
                  <div className="flex items-center space-x-4 mt-1 font-bold text-slate-700">
                    <span className="flex items-center gap-1.5 text-rose-600">
                      <span className="h-2 w-2 rounded-full bg-rose-500"></span>
                      Tinggi: {classDistribution.Tinggi}
                    </span>
                    <span className="flex items-center gap-1.5 text-emerald-600">
                      <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
                      Rendah: {classDistribution.Rendah}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Tabel Utama */}
            <div className="overflow-x-auto rounded-xl border border-slate-150">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-150 text-slate-600 text-xs font-bold">
                    <th className="px-4 py-3">No. Pasien</th>
                    <th className="px-4 py-3">Umur (Tahun)</th>
                    <th className="px-4 py-3">Tekanan Darah (mmHg)</th>
                    <th className="px-4 py-3">Kolesterol (mg/dL)</th>
                    <th className="px-4 py-3">Detak Jantung (bpm)</th>
                    <th className="px-4 py-3">Hasil Risiko Aktual</th>
                  </tr>
                </thead>
                <tbody className="text-xs text-slate-700 divide-y divide-slate-100">
                  {dataset
                    .slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)
                    .map((item, idx) => (
                      <tr key={item.id} className="hover:bg-slate-50/50 transition">
                        <td className="px-4 py-3 font-semibold text-slate-500">Pasien #{item.id}</td>
                        <td className="px-4 py-3 font-semibold">{item.umur} thn</td>
                        <td className="px-4 py-3 font-mono">{item.tekananDarah} mmHg</td>
                        <td className="px-4 py-3 font-mono">{item.kolesterol} mg/dL</td>
                        <td className="px-4 py-3 font-mono">{item.detakJantung} bpm</td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${
                            item.risiko === 'Tinggi' 
                              ? 'bg-rose-50 text-rose-700 border-rose-200' 
                              : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          }`}>
                            <Heart className="h-3 w-3 mr-1 shrink-0" />
                            {item.risiko}
                          </span>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>

            {/* Navigasi Pagination */}
            <div className="flex items-center justify-between pt-4 border-t border-slate-100 text-xs">
              <span className="text-slate-500 font-medium">
                Menampilkan <strong>{Math.min(currentPage * itemsPerPage, dataset.length)}</strong> dari <strong>{dataset.length}</strong> data pasien
              </span>
              <div className="flex space-x-1">
                <button
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                  className="px-2.5 py-1.5 rounded bg-white border border-slate-200 hover:bg-slate-50 text-slate-600 disabled:opacity-40 disabled:hover:bg-white flex items-center gap-1 font-semibold"
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                  Sebelumnya
                </button>
                <button
                  disabled={currentPage >= Math.ceil(dataset.length / itemsPerPage)}
                  onClick={() => setCurrentPage(prev => Math.min(prev + 1, Math.ceil(dataset.length / itemsPerPage)))}
                  className="px-2.5 py-1.5 rounded bg-white border border-slate-200 hover:bg-slate-50 text-slate-600 disabled:opacity-40 disabled:hover:bg-white flex items-center gap-1 font-semibold"
                >
                  Selanjutnya
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ==============================================================================
            TAB: PREDIKSI INTERAKTIF PASIEN BARU
            ============================================================================== */}
        {activeTab === 'predict' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Input Data Pasien Baru */}
            <div className="lg:col-span-5 bg-white p-5 rounded-2xl shadow-sm border border-slate-200/80 space-y-6">
              <div>
                <h3 className="font-bold text-slate-800 text-base flex items-center gap-2 pb-2 border-b border-slate-100">
                  <Sliders className="h-5 w-5 text-indigo-600" />
                  Input Data Klinis Pasien Baru
                </h3>
                <p className="text-[11px] text-slate-500 mt-1">
                  Masukkan parameter indikator medis pasien di bawah ini. Model KNN akan menghitung jarak terdekat ke data riwayat klinis di database dan mengeluarkan hasil klasifikasi saat ini juga.
                </p>
              </div>

              <div className="space-y-4">
                {/* Atribut 1: Umur */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <label className="font-semibold text-slate-600">Umur Pasien (Tahun)</label>
                    <span className="font-mono font-bold text-indigo-700">{inputPasien.umur} tahun</span>
                  </div>
                  <input 
                    type="range" 
                    min="20" 
                    max="90" 
                    value={inputPasien.umur}
                    onChange={(e) => setInputPasien({ ...inputPasien, umur: parseInt(e.target.value) })}
                    className="w-full accent-indigo-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg border-none"
                  />
                  <div className="flex justify-between text-[9px] text-slate-400">
                    <span>Min: 20 thn</span>
                    <span>Max: 90 thn</span>
                  </div>
                </div>

                {/* Atribut 2: Tekanan Darah */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <label className="font-semibold text-slate-600">Tekanan Darah (Sistolik - mmHg)</label>
                    <span className="font-mono font-bold text-indigo-700">{inputPasien.tekananDarah} mmHg</span>
                  </div>
                  <input 
                    type="range" 
                    min="80" 
                    max="200" 
                    value={inputPasien.tekananDarah}
                    onChange={(e) => setInputPasien({ ...inputPasien, tekananDarah: parseInt(e.target.value) })}
                    className="w-full accent-indigo-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg border-none"
                  />
                  <div className="flex justify-between text-[9px] text-slate-400">
                    <span>Sangat Rendah: 80</span>
                    <span>Krisis Hipertensi: 200</span>
                  </div>
                </div>

                {/* Atribut 3: Kolesterol */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <label className="font-semibold text-slate-600">Tingkat Kolesterol (mg/dL)</label>
                    <span className="font-mono font-bold text-indigo-700">{inputPasien.kolesterol} mg/dL</span>
                  </div>
                  <input 
                    type="range" 
                    min="100" 
                    max="350" 
                    value={inputPasien.kolesterol}
                    onChange={(e) => setInputPasien({ ...inputPasien, kolesterol: parseInt(e.target.value) })}
                    className="w-full accent-indigo-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg border-none"
                  />
                  <div className="flex justify-between text-[9px] text-slate-400">
                    <span>Ideal: 100</span>
                    <span>Sangat Tinggi: 350</span>
                  </div>
                </div>

                {/* Atribut 4: Detak Jantung */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <label className="font-semibold text-slate-600">Detak Jantung Istirahat (bpm)</label>
                    <span className="font-mono font-bold text-indigo-700">{inputPasien.detakJantung} bpm</span>
                  </div>
                  <input 
                    type="range" 
                    min="50" 
                    max="150" 
                    value={inputPasien.detakJantung}
                    onChange={(e) => setInputPasien({ ...inputPasien, detakJantung: parseInt(e.target.value) })}
                    className="w-full accent-indigo-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg border-none"
                  />
                  <div className="flex justify-between text-[9px] text-slate-400">
                    <span>Lambat (Bradycardia): 50</span>
                    <span>Cepat (Tachycardia): 150</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Hasil Prediksi Pasien Baru */}
            <div className="lg:col-span-7 bg-white p-5 rounded-2xl shadow-sm border border-slate-200/80 flex flex-col justify-between space-y-6">
              
              {/* Hasil Ringkas Diagnosa */}
              <div className="space-y-4">
                <h3 className="font-bold text-slate-800 text-base pb-2 border-b border-slate-100">
                  Hasil Prediksi Real-Time Model KNN
                </h3>

                {hasilPrediksiPasien && (
                  <div className={`p-5 rounded-2xl border ${
                    hasilPrediksiPasien.prediction === 'Tinggi' 
                      ? 'bg-rose-50 border-rose-200 text-rose-900' 
                      : 'bg-emerald-50 border-emerald-200 text-emerald-900'
                  }`}>
                    <div className="flex items-center space-x-3 mb-3">
                      {hasilPrediksiPasien.prediction === 'Tinggi' ? (
                        <div className="bg-rose-600 text-white p-1.5 rounded-full">
                          <ShieldAlert className="h-6 w-6 animate-pulse" />
                        </div>
                      ) : (
                        <div className="bg-emerald-600 text-white p-1.5 rounded-full">
                          <CheckCircle2 className="h-6 w-6" />
                        </div>
                      )}
                      <div>
                        <span className="text-[10px] uppercase font-bold tracking-wider opacity-60">Status Prediksi Risiko</span>
                        <h4 className="text-xl font-black leading-tight">
                          RISIKO {hasilPrediksiPasien.prediction.toUpperCase()}
                        </h4>
                      </div>
                    </div>

                    <p className="text-xs leading-relaxed opacity-90">
                      {hasilPrediksiPasien.prediction === 'Tinggi' 
                        ? 'Pasien ini memiliki kombinasi profil klinis yang berada sangat dekat dengan kelompok penderita penyakit jantung. Sangat disarankan untuk segera melakukan konsultasi menyeluruh dengan dokter spesialis jantung.' 
                        : 'Hasil analisis menunjukkan profil klinis pasien ini berada dalam batas aman dan berkelompok di area riwayat medis risiko rendah. Pertahankan gaya hidup sehat!'
                      }
                    </p>

                    <div className="mt-4 pt-4 border-t border-slate-200/40 flex items-center justify-between text-xs font-semibold">
                      <span>Tingkat Keyakinan Model:</span>
                      <span className="font-mono text-sm bg-white/70 px-2 py-0.5 rounded-md border">
                        {hasilPrediksiPasien.confidence.toFixed(0)}% suara tetangga terdekat
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Rincian Struktur Tetangga Terdekat (K-Nearest Neighbors Detail) */}
              <div className="space-y-3">
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide">
                  Daftar 3 Tetangga Terdekat ({kValue}-Nearest Neighbors Terpilih)
                </h4>
                
                {hasilPrediksiPasien && (
                  <div className="space-y-2">
                    {hasilPrediksiPasien.neighbors.map((neighbor, idx) => {
                      // Ambil data asli dari dataset berdasarkan indeks atau lakukan pencocokan terdekat
                      const isTinggi = neighbor.label === 'Tinggi';
                      return (
                        <div 
                          key={idx}
                          className="flex items-center justify-between p-3 bg-slate-50 rounded-xl border border-slate-150 text-xs"
                        >
                          <div className="flex items-center space-x-3">
                            <span className="font-mono text-[10px] font-bold text-slate-400 bg-slate-200/50 h-5 w-5 rounded-full flex items-center justify-center">
                              #{idx + 1}
                            </span>
                            <div>
                              <span className="font-bold text-slate-700 block">Jarak Euclidean: <span className="font-mono text-indigo-700">{neighbor.distance.toFixed(3)}</span></span>
                              <span className="text-[10px] text-slate-400">Profil medis tetangga terdekat di database data latih.</span>
                            </div>
                          </div>
                          
                          <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${
                            isTinggi 
                              ? 'bg-rose-50 text-rose-700 border-rose-100' 
                              : 'bg-emerald-50 text-emerald-700 border-emerald-100'
                          }`}>
                            {neighbor.label}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

            </div>
          </div>
        )}

        {/* ==============================================================================
            TAB: MATERI TEORI DASAR KNN
            ============================================================================== */}
        {activeTab === 'theory' && (
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80 space-y-6">
            <div className="border-b border-slate-100 pb-4">
              <h3 className="font-bold text-slate-800 text-lg flex items-center gap-2">
                <BookOpen className="h-5 w-5 text-indigo-600" />
                Dasar Teori Algoritma K-Nearest Neighbor (KNN)
              </h3>
              <p className="text-xs text-slate-500 mt-1">Edukasi komprehensif bagi mahasiswa mengenai fondasi matematika di balik klasifikasi data klinis.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-600 leading-relaxed">
              <div className="space-y-4">
                <h4 className="font-bold text-slate-800 text-sm flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 bg-indigo-600 rounded-full"></span>
                  1. Bagaimana KNN Bekerja?
                </h4>
                <p className="text-xs">
                  KNN adalah algoritma <strong>supervised machine learning</strong> non-parametrik yang melakukan klasifikasi berdasarkan kemiripan fitur. Saat data baru dimasukkan, algoritma tidak memproses permodelan abstrak, melainkan mencari database riwayat secara langsung untuk membandingkan karakteristik pasien baru tersebut dengan pasien-pasien terdahulu yang sudah diketahui tingkat risikonya.
                </p>

                <h4 className="font-bold text-slate-800 text-sm flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 bg-indigo-600 rounded-full"></span>
                  2. Konsep Jarak Euclidean
                </h4>
                <p className="text-xs">
                  KNN mengukur kedekatan antar pasien menggunakan persamaan geometri jarak garis lurus (Euclidean). Jika $x$ adalah pasien baru dan $y$ adalah pasien di database, dengan $n$ parameter medis (Umur, Tensi, dsb), formulanya adalah:
                </p>
                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100 font-mono text-xs text-center text-slate-700">
                  d(x, y) = √ Σ (x_i - y_i)²
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-bold text-slate-800 text-sm flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 bg-indigo-600 rounded-full"></span>
                  3. Mengapa Standardisasi Skala Begitu Penting?
                </h4>
                <p className="text-xs">
                  Dalam data medis, rentang nilai antar atribut berbeda jauh. Kolesterol bernilai ratusan (150-320 mg/dL), sedangkan umur bernilai puluhan (30-74 tahun). Tanpa standardisasi, perbedaan nilai kolesterol akan mendominasi hasil kuadrat jarak Euclidean secara mutlak, sehingga variabel umur menjadi tidak berpengaruh.
                </p>
                <p className="text-xs">
                  <strong>StandardScaler</strong> memetakan ulang seluruh fitur agar memiliki nilai rata-rata (*mean*) = 0 dan variansi standar (*standard deviation*) = 1 menggunakan rumus:
                </p>
                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100 font-mono text-xs text-center text-slate-700">
                  z = (x - μ) / σ
                </div>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* ==============================================================================
          FOOTER
          ============================================================================== */}
      <footer className="bg-slate-900 text-slate-400 text-center py-4 text-xs border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-2">
          <p>© 2026 CardioKNN. Program Pendukung Praktikum Data Science Mahasiswa.</p>
          <div className="flex items-center space-x-3 text-[10px]">
            <span className="bg-slate-800 px-2 py-0.5 rounded text-slate-300">Python 3</span>
            <span className="bg-slate-800 px-2 py-0.5 rounded text-slate-300">scikit-learn</span>
            <span className="bg-slate-800 px-2 py-0.5 rounded text-slate-300">React Tail</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
