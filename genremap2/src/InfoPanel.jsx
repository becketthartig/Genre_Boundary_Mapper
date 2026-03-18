// InfoPanel.jsx — contextual info panel shown below the header
import { useState } from 'react'
import styles from './InfoPanel.module.css'

const FEATURE_GROUPS = [
  {
    group: 'Rhythm',
    features: [
      { name: 'tempo', desc: 'Beats per minute — how fast the song is' },
      { name: 'zero_crossing_rate_mean', desc: 'How often the audio signal crosses zero — higher in noisy or percussive sounds like hi-hats' },
      { name: 'zero_crossing_rate_var', desc: 'How much the zero crossing rate varies throughout the song' },
    ]
  },
  {
    group: 'Energy & Loudness',
    features: [
      { name: 'rms_mean', desc: 'Average loudness/energy of the song (Root Mean Square of the waveform)' },
      { name: 'rms_var', desc: 'How much the loudness fluctuates — high in dynamic songs, low in consistent ones' },
      { name: 'harmony_mean', desc: 'Average energy in the harmonic (tonal/melodic) part of the signal' },
      { name: 'harmony_var', desc: 'How much the harmonic content varies throughout the song' },
      { name: 'perceptr_mean', desc: 'Average energy in the percussive part of the signal (drums, hits)' },
      { name: 'perceptr_var', desc: 'How much the percussive content varies' },
    ]
  },
  {
    group: 'Spectral Shape',
    features: [
      { name: 'spectral_centroid_mean', desc: 'Average "brightness" — where the center of mass of the frequency spectrum sits. High = bright/trebly (metal, pop), low = dark/bassy (blues, jazz)' },
      { name: 'spectral_centroid_var', desc: 'How much the brightness changes over time' },
      { name: 'spectral_bandwidth_mean', desc: 'Average spread of frequencies around the centroid — wider means more complex, fuller sound' },
      { name: 'spectral_bandwidth_var', desc: 'How much the frequency spread varies' },
      { name: 'rolloff_mean', desc: 'The frequency below which 85% of the signal energy sits — higher rolloff = brighter sound' },
      { name: 'rolloff_var', desc: 'How much the rolloff frequency varies over the song' },
      { name: 'chroma_stft_mean', desc: 'Average distribution of energy across the 12 musical pitch classes (C, C#, D...) — captures harmony and key' },
      { name: 'chroma_stft_var', desc: 'How much the pitch class distribution changes — high in harmonically varied or modulating songs' },
    ]
  },
  {
    group: 'Timbre — MFCCs (mfcc1–mfcc20, mean & var)',
    description: 'Mel-Frequency Cepstral Coefficients capture the texture and timbre of sound — what makes a guitar sound different from a violin even on the same note. Each coefficient captures a different layer of tonal character. Lower-numbered MFCCs describe broad tonal shape; higher-numbered ones describe finer texture. Each has a mean (average value across the song) and var (how much it fluctuates). These 40 values (20 × mean/var) are the most powerful features for genre classification.',
    features: [
      { name: 'mfcc1_mean / mfcc1_var', desc: 'Overall loudness/energy of the spectrum — closely related to volume' },
      { name: 'mfcc2_mean / mfcc2_var', desc: 'Broad spectral shape — distinguishes bright vs dark timbres' },
      { name: 'mfcc3–5_mean / var', desc: 'Mid-level tonal characteristics — captures vowel-like qualities of instruments' },
      { name: 'mfcc6–10_mean / var', desc: 'Mid-level timbral texture — instrument body resonances' },
      { name: 'mfcc11–20_mean / var', desc: 'Fine-grained timbral detail — subtle texture differences between instruments and production styles' },
    ]
  }
]

const VIZ_INFO = {
  scatter: {
    title: 'Boundary Exploration Map',
    body: `This map uses UMAP (Uniform Manifold Approximation and Projection) — an unsupervised dimensionality reduction technique — to project all 57 audio features down to 2 dimensions. Unlike supervised methods, UMAP preserves the natural perceptual structure of the music without being biased by genre labels, revealing how songs actually sound rather than how they're categorized.

Each point is a song. Songs close together share similar acoustic characteristics across the 57-dimensional feature space. UMAP preserves both local neighborhoods (songs that sound similar) and global structure (overall clusters and patterns), making it excellent for discovering unexpected similarities and exploring the continuous spectrum of musical styles.

The boundary filter on the right lets you show only songs near genre boundaries — songs that are acoustically ambiguous because they're closer to a different genre than to their own labeled genre in feature space.`,
  },
  violin: {
    title: 'Feature by Genre',
    body: `This view shows how a single audio feature is distributed across each genre. Select any feature from the pill buttons at the top.

Each box plot shows:
• The box = the middle 50% of songs (Q1 to Q3). Tall box = inconsistent genre, short box = consistent.
• The horizontal line = median (middle value).
• The dot = mean (average). If far from the median, the distribution is skewed.
• The whiskers = min and max values, excluding extreme outliers.

Use this to find features that best separate genres — if two genres have non-overlapping boxes, that feature is a strong classifier signal.`,
  },
  heatmap: {
    title: 'Feature Correlation',
    body: `This heatmap shows the Pearson correlation coefficient (r) between every pair of audio features across all songs.

• r = +1 (teal): features always move together — adding both gives redundant information
• r = -1 (blue): features move in opposite directions
• r = 0 (dark): no relationship

Highly correlated features are essentially measuring the same thing. This is useful for understanding the feature space and deciding which features to include or drop when recomputing the projection.

Sort by Variance to bring the most variable (and usually most informative) features to the top.`,
  },
  umap3d: {
    title: 'UMAP Projection',
    body: `UMAP (Uniform Manifold Approximation and Projection) compresses the 57 audio features down to 2 or 3 dimensions while trying to preserve local similarity structure. Unlike LDA, UMAP is unsupervised — it ignores genre labels entirely and purely optimizes for keeping acoustically similar songs close together.

This means the clusters you see here are based purely on how songs sound, not on their labels. You may see unexpected neighbors: a jazz song next to a blues song, or a pop song next to disco, because they genuinely share acoustic characteristics.

Use the 2D view for a quick overview. Use the 3D view and rotate it to see structure that the 2D projection might flatten out.`,
  },
}

export default function InfoPanel({ activeViz, onClose }) {
  const [tab, setTab] = useState('about')
  const vizInfo = VIZ_INFO[activeViz] || VIZ_INFO['scatter']

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${tab === 'about' ? styles.tabActive : ''}`}
            onClick={() => setTab('about')}
          >
            About this view
          </button>
          <button
            className={`${styles.tab} ${tab === 'features' ? styles.tabActive : ''}`}
            onClick={() => setTab('features')}
          >
            Feature Glossary
          </button>
        </div>
        <button className={styles.closeBtn} onClick={onClose}>✕</button>
      </div>

      <div className={styles.panelBody}>
        {tab === 'about' && (
          <div className={styles.aboutContent}>
            <div className={styles.aboutTitle}>{vizInfo.title}</div>
            {vizInfo.body.split('\n\n').map((para, i) => (
              <p key={i} className={styles.aboutPara}>{para}</p>
            ))}
          </div>
        )}

        {tab === 'features' && (
          <div className={styles.glossary}>
            {FEATURE_GROUPS.map(group => (
              <div key={group.group} className={styles.glossaryGroup}>
                <div className={styles.glossaryGroupName}>{group.group}</div>
                {group.description && (
                  <div className={styles.glossaryGroupDesc}>{group.description}</div>
                )}
                {group.features.map(f => (
                  <div key={f.name} className={styles.glossaryRow}>
                    <span className={styles.glossaryFeature}>{f.name}</span>
                    <span className={styles.glossaryDesc}>{f.desc}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}