import { useState } from 'react';
import axios from 'axios';
import './MixDesignForm.css';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

const INITIAL_STATE = {
  cement: '',
  silica_fume: '',
  fly_ash: '',
  sand: '',
  coarse_aggregate: '',
  water: '',
  superplasticizer: '',
  fiber_type: 'steel',
  fiber_content_percent: '',
  water_binder_ratio: '',
  curing_age_days: '',
  curing_temp_celsius: '',
  specimen_type: 'cylinder',
};

const NUMERIC_FIELDS = [
  {
    id: 'cement',
    label: 'Cement',
    unit: 'kg/m³',
    placeholder: 'e.g. 800',
    min: 0,
    step: 1,
  },
  {
    id: 'silica_fume',
    label: 'Silica Fume',
    unit: 'kg/m³',
    placeholder: 'e.g. 150',
    min: 0,
    step: 0.1,
  },
  {
    id: 'fly_ash',
    label: 'Fly Ash',
    unit: 'kg/m³',
    placeholder: 'e.g. 0',
    min: 0,
    step: 0.1,
  },
  {
    id: 'sand',
    label: 'Sand',
    unit: 'kg/m³',
    placeholder: 'e.g. 1000',
    min: 0,
    step: 1,
  },
  {
    id: 'coarse_aggregate',
    label: 'Coarse Aggregate',
    unit: 'kg/m³',
    placeholder: 'e.g. 0',
    min: 0,
    step: 1,
  },
  {
    id: 'water',
    label: 'Water',
    unit: 'kg/m³',
    placeholder: 'e.g. 140',
    min: 0,
    step: 0.1,
  },
  {
    id: 'superplasticizer',
    label: 'Superplasticizer',
    unit: 'kg/m³',
    placeholder: 'e.g. 30',
    min: 0,
    step: 0.1,
  },
  {
    id: 'fiber_content_percent',
    label: 'Fiber Content',
    unit: '%vol',
    placeholder: 'e.g. 2.0',
    min: 0,
    max: 10,
    step: 0.1,
  },
  {
    id: 'water_binder_ratio',
    label: 'Water / Binder Ratio',
    unit: '—',
    placeholder: 'e.g. 0.18',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    id: 'curing_age_days',
    label: 'Curing Age',
    unit: 'days',
    placeholder: 'e.g. 28',
    min: 0,
    step: 1,
  },
  {
    id: 'curing_temp_celsius',
    label: 'Curing Temperature',
    unit: '°C',
    placeholder: 'e.g. 20',
    min: -10,
    max: 300,
    step: 1,
  },
];

const FIBER_TYPES = ['steel', 'polyethylene', 'polypropylene', 'basalt', 'carbon', 'none'];
const SPECIMEN_TYPES = ['cylinder', 'cube', 'prism'];

export default function MixDesignForm() {
  const [formData, setFormData] = useState(INITIAL_STATE);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleChange(e) {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  }

  function handleReset() {
    setFormData(INITIAL_STATE);
    setPrediction(null);
    setError(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setError(null);

    // Coerce numeric string fields to numbers before sending
    const NUMERIC_KEYS = [
      'cement', 'silica_fume', 'fly_ash', 'sand', 'coarse_aggregate',
      'water', 'superplasticizer', 'fiber_content_percent',
      'water_binder_ratio', 'curing_age_days', 'curing_temp_celsius',
    ];
    const payload = { ...formData };
    for (const key of NUMERIC_KEYS) {
      payload[key] = payload[key] === '' ? null : Number(payload[key]);
    }

    try {
      const { data } = await axios.post(`${API_URL}/predict`, payload, {
        headers: { 'Content-Type': 'application/json' },
      });
      setPrediction({
        strength: data.predicted_strength_MPa,
        shap: data.shap_contributions ?? null,
      });
    } catch (err) {
      const message =
        err.response?.data?.detail ??
        err.response?.data?.message ??
        err.message ??
        'Unknown error';
      setError(String(message));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mdf-wrapper">
      <header className="mdf-header">
        <h1 className="mdf-title">UHPFRC Strength Predictor</h1>
        <p className="mdf-subtitle">
          Enter your mix-design parameters to predict compressive strength.
        </p>
      </header>

      <form className="mdf-form" onSubmit={handleSubmit} noValidate>
        {/* ── Binder Materials ────────────────────────────── */}
        <fieldset className="mdf-fieldset">
          <legend className="mdf-legend">Binder Materials</legend>
          <div className="mdf-grid">
            {['cement', 'silica_fume', 'fly_ash'].map((id) => {
              const field = NUMERIC_FIELDS.find((f) => f.id === id);
              return (
                <div className="mdf-field" key={id}>
                  <label htmlFor={id} className="mdf-label">
                    {field.label}
                    <span className="mdf-unit">{field.unit}</span>
                  </label>
                  <input
                    id={id}
                    name={id}
                    type="number"
                    className="mdf-input"
                    value={formData[id]}
                    onChange={handleChange}
                    placeholder={field.placeholder}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                  />
                </div>
              );
            })}
          </div>
        </fieldset>

        {/* ── Aggregates ──────────────────────────────────── */}
        <fieldset className="mdf-fieldset">
          <legend className="mdf-legend">Aggregates</legend>
          <div className="mdf-grid">
            {['sand', 'coarse_aggregate'].map((id) => {
              const field = NUMERIC_FIELDS.find((f) => f.id === id);
              return (
                <div className="mdf-field" key={id}>
                  <label htmlFor={id} className="mdf-label">
                    {field.label}
                    <span className="mdf-unit">{field.unit}</span>
                  </label>
                  <input
                    id={id}
                    name={id}
                    type="number"
                    className="mdf-input"
                    value={formData[id]}
                    onChange={handleChange}
                    placeholder={field.placeholder}
                    min={field.min}
                    step={field.step}
                  />
                </div>
              );
            })}
          </div>
        </fieldset>

        {/* ── Liquids & Admixtures ────────────────────────── */}
        <fieldset className="mdf-fieldset">
          <legend className="mdf-legend">Liquids &amp; Admixtures</legend>
          <div className="mdf-grid">
            {['water', 'superplasticizer', 'water_binder_ratio'].map((id) => {
              const field = NUMERIC_FIELDS.find((f) => f.id === id);
              return (
                <div className="mdf-field" key={id}>
                  <label htmlFor={id} className="mdf-label">
                    {field.label}
                    <span className="mdf-unit">{field.unit}</span>
                  </label>
                  <input
                    id={id}
                    name={id}
                    type="number"
                    className="mdf-input"
                    value={formData[id]}
                    onChange={handleChange}
                    placeholder={field.placeholder}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                  />
                </div>
              );
            })}
          </div>
        </fieldset>

        {/* ── Fiber Reinforcement ─────────────────────────── */}
        <fieldset className="mdf-fieldset">
          <legend className="mdf-legend">Fiber Reinforcement</legend>
          <div className="mdf-grid">
            <div className="mdf-field">
              <label htmlFor="fiber_type" className="mdf-label">
                Fiber Type
              </label>
              <select
                id="fiber_type"
                name="fiber_type"
                className="mdf-input mdf-select"
                value={formData.fiber_type}
                onChange={handleChange}
              >
                {FIBER_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </option>
                ))}
              </select>
            </div>
            <div className="mdf-field">
              <label htmlFor="fiber_content_percent" className="mdf-label">
                Fiber Content
                <span className="mdf-unit">%vol</span>
              </label>
              <input
                id="fiber_content_percent"
                name="fiber_content_percent"
                type="number"
                className="mdf-input"
                value={formData.fiber_content_percent}
                onChange={handleChange}
                placeholder="e.g. 2.0"
                min={0}
                max={10}
                step={0.1}
              />
            </div>
          </div>
        </fieldset>

        {/* ── Curing & Specimen ───────────────────────────── */}
        <fieldset className="mdf-fieldset">
          <legend className="mdf-legend">Curing &amp; Specimen</legend>
          <div className="mdf-grid">
            {['curing_age_days', 'curing_temp_celsius'].map((id) => {
              const field = NUMERIC_FIELDS.find((f) => f.id === id);
              return (
                <div className="mdf-field" key={id}>
                  <label htmlFor={id} className="mdf-label">
                    {field.label}
                    <span className="mdf-unit">{field.unit}</span>
                  </label>
                  <input
                    id={id}
                    name={id}
                    type="number"
                    className="mdf-input"
                    value={formData[id]}
                    onChange={handleChange}
                    placeholder={field.placeholder}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                  />
                </div>
              );
            })}
            <div className="mdf-field">
              <label htmlFor="specimen_type" className="mdf-label">
                Specimen Type
              </label>
              <select
                id="specimen_type"
                name="specimen_type"
                className="mdf-input mdf-select"
                value={formData.specimen_type}
                onChange={handleChange}
              >
                {SPECIMEN_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </fieldset>

        {/* ── Actions ─────────────────────────────────────── */}
        <div className="mdf-actions">
          <button
            type="button"
            id="btn-reset"
            className="mdf-btn mdf-btn--secondary"
            onClick={handleReset}
            disabled={loading}
          >
            Reset
          </button>
          <button
            type="submit"
            id="btn-predict"
            className="mdf-btn mdf-btn--primary"
            disabled={loading}
          >
            {loading ? 'Predicting…' : 'Predict Strength'}
          </button>
        </div>
      </form>

      {/* ── Error banner ────────────────────────────────── */}
      {error && (
        <div id="result-error" className="mdf-result mdf-result--error" role="alert">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* ── Result card ─────────────────────────────────── */}
      {prediction && (
        <div id="result-card" className="mdf-result mdf-result--success">
          <p className="mdf-result-label">Predicted Compressive Strength</p>
          <p className="mdf-result-value">
            {Number(prediction.strength).toFixed(2)}
            <span className="mdf-result-unit"> MPa</span>
          </p>

          {prediction.shap && Object.keys(prediction.shap).length > 0 && (
            <details className="mdf-shap">
              <summary className="mdf-shap-summary">SHAP Contributions</summary>
              <ul className="mdf-shap-list">
                {Object.entries(prediction.shap)
                  .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                  .map(([feature, value]) => (
                    <li key={feature} className="mdf-shap-item">
                      <span className="mdf-shap-feature">{feature}</span>
                      <span
                        className={`mdf-shap-val ${Number(value) >= 0 ? 'mdf-shap-pos' : 'mdf-shap-neg'}`}
                      >
                        {Number(value) >= 0 ? '+' : ''}{Number(value).toFixed(3)}
                      </span>
                    </li>
                  ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
