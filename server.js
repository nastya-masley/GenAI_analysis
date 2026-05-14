const path = require('path');
const fs = require('fs');
const express = require('express');
const multer = require('multer');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();

const PORT = process.env.PORT || 3000;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const MAX_VIDEO_SIZE_MB = Number(process.env.MAX_VIDEO_SIZE_MB) || 200;

const DEFAULT_PROMPT = `You are an expert in nonverbal communication, emotion analysis and human behavior.

Analyze this video. Focus on emotions, facial expressions, posture and gestures. Be concise.

FORMATTING RULES:
- Do NOT start with filler phrases like "Sure!", "Here's...", "Certainly!", etc. Start directly with the analysis.
- Use numbered sections (0. 1. 2. 3. 4.) and subsections (1.1 1.2 etc.) as headings.
- Use * for bullet points, each on its own line.
- Use **bold** for key terms and emotions.
- Separate major sections with --- on its own line.
- Keep blank lines between sections and subsections.
- Each bullet: max 1-2 short sentences.
- Do NOT invent details. If something cannot be assessed, write: "Not enough visual data."

Structure:

0. Overall Picture
- One sentence describing what is happening in the video.

1. Emotional Analysis
1.1 Emotional Tone
- Dominant emotions.
- Valence: positive / neutral / negative.
1.2 Emotion Dynamics
- How emotions shift from start to middle to end.
- Any sharp emotional changes.
1.3 Incongruence
- Mismatch between verbal context and nonverbal signals.

2. Facial Analysis
2.1 People
- How many visible. Label as Person 1, Person 2, etc.
2.2 Expressions
- Main emotions via facial expression per person.
- Micro-expressions if noticeable.
2.3 Gaze
- Eye contact with camera or others.
- Gaze aversion direction and meaning.

3. Body and Gestures
3.1 Posture
- Open vs closed. Body tension level.
3.2 Gestures
- Hand gestures: controlled / natural / excessive.
- Self-soothing gestures (touching neck, face, hands).
3.3 Space
- Distance to others or camera. Leaning direction.

4. Summary
4.1 Profile
- 3-5 bullets on emotional state, confidence, engagement.
4.2 Key Signals
- 3-5 most important nonverbal signals with short explanations.
4.3 Interpretation
- What the nonverbal behavior indicates (trust, stress, confidence, defensiveness, etc.).`;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_VIDEO_SIZE_MB * 1024 * 1024
  }
});

if (!GEMINI_API_KEY) {
  console.warn('Warning: GEMINI_API_KEY is not set. /api/analyze requests will fail.');
}

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));
app.use('/assets', express.static(path.join(__dirname, 'assets')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

const framesDir = path.join(__dirname, 'assets', 'export', 'frames');
fs.mkdirSync(framesDir, { recursive: true });

const libraryDir = path.join(__dirname, 'assets', 'archive', 'library');
fs.mkdirSync(libraryDir, { recursive: true });

const LIBRARY_EXTS = new Set(['.mp4', '.mov', '.webm', '.jpg', '.jpeg', '.png']);
const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png']);

app.get('/api/library', (_req, res) => {
  try {
    const files = fs.readdirSync(libraryDir).filter((f) => {
      const ext = path.extname(f).toLowerCase();
      return LIBRARY_EXTS.has(ext) && !f.startsWith('.');
    });
    const items = files.map((f) => {
      const ext = path.extname(f).toLowerCase();
      return {
        name: f,
        path: `/assets/archive/library/${encodeURIComponent(f)}`,
        type: IMAGE_EXTS.has(ext) ? 'image' : 'video',
      };
    });
    res.json(items);
  } catch (err) {
    res.status(500).json({ error: 'Failed to read library' });
  }
});

app.post('/api/capture-frame', express.raw({ type: 'image/png', limit: '20mb' }), (req, res) => {
  const filename = req.query.filename;
  if (!filename || !/^frame_.+_\d{2}-\d{2}\.png$/.test(filename)) {
    return res.status(400).json({ error: 'Invalid filename' });
  }
  const safeName = path.basename(filename);
  const ext = path.extname(safeName);
  const stem = safeName.slice(0, -ext.length);
  let finalName = safeName;
  let n = 1;
  while (fs.existsSync(path.join(framesDir, finalName))) {
    finalName = `${stem} (copy ${n})${ext}`;
    n++;
  }
  fs.writeFileSync(path.join(framesDir, finalName), req.body);
  res.json({ ok: true, name: finalName, path: `/assets/export/frames/${encodeURIComponent(finalName)}` });
});

app.post('/api/capture-frameset-frame-v2', express.raw({ type: 'image/png', limit: '20mb' }), (req, res) => {
  const { dir, filename } = req.query;
  if (!dir || !filename) return res.status(400).json({ error: 'Missing dir or filename' });
  const safeFile = path.basename(filename);
  const target = path.isAbsolute(dir) ? dir : path.join(__dirname, dir);
  try {
    fs.mkdirSync(target, { recursive: true });
    fs.writeFileSync(path.join(target, safeFile), req.body);
    res.json({ ok: true, path: path.join(target, safeFile) });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/archive-clip', express.raw({ type: 'video/webm', limit: '50mb' }), (req, res) => {
  const now = new Date();
  const ts = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    '_',
    String(now.getHours()).padStart(2, '0'),
    '-',
    String(now.getMinutes()).padStart(2, '0'),
    '-',
    String(now.getSeconds()).padStart(2, '0'),
  ].join('');
  const filename = `exhibition_${ts}.webm`;
  const filePath = path.join(libraryDir, filename);
  fs.writeFileSync(filePath, req.body);
  res.json({ ok: true, name: filename, path: `/assets/archive/library/${encodeURIComponent(filename)}` });
});

app.post('/api/analyze', upload.single('video'), async (req, res, next) => {
  try {
    if (!GEMINI_API_KEY) {
      return res.status(500).json({ error: 'Server misconfiguration: missing GEMINI_API_KEY.' });
    }

    const promptInput = req.body.prompt?.trim();
    const prompt = promptInput || DEFAULT_PROMPT;

    const videoFile = req.file;
    if (!videoFile) {
      return res.status(400).json({ error: 'File is required.' });
    }

    const base64Video = videoFile.buffer.toString('base64');
    const mimeType = videoFile.mimetype || 'video/mp4';

    const payload = {
      contents: [
        {
          role: 'user',
          parts: [
            { text: prompt },
            {
              inlineData: {
                mimeType,
                data: base64Video
              }
            }
          ]
        }
      ]
    };

    const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;
    const geminiResponse = await fetch(geminiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!geminiResponse.ok) {
      const errorText = await geminiResponse.text();
      let geminiMessage = errorText;
      let rawResponse = errorText;
      try {
        const parsed = JSON.parse(errorText);
        geminiMessage = parsed?.error?.message || errorText;
        rawResponse = JSON.stringify(parsed, null, 2);
      } catch (_) {}
      console.error(`[analyze] Gemini error — HTTP ${geminiResponse.status}: ${geminiMessage}`);
      console.error(`[analyze] Full Gemini response:\n${rawResponse}`);
      return res.status(502).json({
        error: `Gemini API error (${geminiResponse.status}): ${geminiMessage}`,
        geminiResponse: rawResponse
      });
    }

    const result = await geminiResponse.json();
    const output = [];
    if (Array.isArray(result?.candidates)) {
      result.candidates.forEach((candidate) => {
        candidate?.content?.parts?.forEach((part) => {
          if (part?.text) {
            output.push(part.text);
          }
        });
      });
    }

    res.json({
      resultText: output.join('\n\n') || 'AI did not return any text.',
      raw: result
    });
  } catch (err) {
    next(err);
  }
});

app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.use((err, _req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res
        .status(413)
        .json({ error: `Video is too large. Max supported size is ${MAX_VIDEO_SIZE_MB} MB.` });
    }
    return res.status(400).json({ error: `Upload failed: ${err.message}` });
  }

  if (err) {
    console.error('Unhandled error:', err);
    return res.status(500).json({ error: 'Internal server error', details: err.message });
  }

  next();
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});


