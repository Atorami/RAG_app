const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

app.post('/api/ask', async (req, res) => {
  console.log('req.body:', req.body);
  try {
    const query = req.body.query;
    if (!query) {
      return res.status(400).json({ error: 'Missing query in body' });
    }
    const response = await axios.post('http://localhost:8000/rag', { query });
    res.json(response.data);
  } catch (err) {
    console.error('Error in /api/ask:', err.message);
    res.status(500).json({ error: 'Error contacting RAG backend' });
  }
});


app.listen(3000, () => {
  console.log('Express API listening on http://localhost:3000');
});
