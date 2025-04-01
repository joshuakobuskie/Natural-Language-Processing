const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');

// Initialize Express
const app = express();
const port = 5001;

// Middleware
app.use(cors());
app.use(express.json());

app.post("/api/generate", async (req, res) => {

  var dataToSend = "";
  const process = spawn("python3", ["./temp.py", req.body.prompt, JSON.stringify(req.body.rag), JSON.stringify(req.body.history)] ); 

  process.stdout.on("data", (data) => {
    dataToSend = data.toString();
  });

  process.stderr.on("data", (data) => {
    console.error("Error from Python:", data.toString());
  });

  process.on("close", (code) => {
    if (!dataToSend) {
      res.status(500).send("Server Connection Error");
    } else {
      res.json({ response: dataToSend });
    }
  });

  process.on("error", (err) => {
    console.error("Failed to start process:", err);
    res.status(500).send("Server Connection Error");
  });
});

// Start server
app.listen(port, () => {
  console.log(`Node.js server running on http://http://192.168.1.163:${port}`);
});

// require('dotenv').config(); // Load environment variables
// const { GoogleGenerativeAI } = require('@google/generative-ai');

// Old implementation, replaced with call to python file
// // Initialize Gemini
// const genAI = new GoogleGenerativeAI(process.env.API_KEY); // Use environment variable
// const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

// // API endpoint
// app.post('/api/generate', async (req, res) => {
//   const { prompt } = req.body;

//   // Should never occur because the frontend stops this behavior
//   if (!prompt) {
//     return res.status(400).json({ error: 'Prompt is required' });
//   }

//   try {
//     // Call Gemini API
//     const result = await model.generateContent(prompt);
//     const response = await result.response;
//     const text = response.text();

//     res.json({ response: text });

//   } catch (error) {
//     console.error('Error:', error);
//     res.status(500).json({ error: 'Failed to generate response' });
//   }
// });