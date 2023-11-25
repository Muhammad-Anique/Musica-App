const express = require('express');
const { exec } = require('child_process');

const cors = require('cors');

const app = express();

// Enable CORS for all routes
app.use(cors());
app.use(express.json()); // Add this line before your routes to parse JSON



const PORT = 8080; // You can use any port number you prefer
const pythonScriptPath = 'C:\\Users\\hp\\Desktop\\musicc\\server\\recommend.py';
const fs = require('fs');
const csv = require('csv-parser');



app.get('/', (req, res) => {
  exec(`python ${pythonScriptPath}`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing Python script: ${error}`);
      res.send("Error")
      return;
    }
    console.log(`Python script executed successfully: ${stdout}`);
    res.send("Success")
  });
});



app.get('/getr', (req, res) => {
  console.log("GETRECOMMENDATIONS")
  const result = []
  fs.createReadStream('C:\\Users\\hp\\Desktop\\musicc\\server\\output.csv')
  .pipe(csv())
  .on('data', (data) => {
    result.push(data)
    
  })
  .on('end', () => {
    res.send(result)
    console.log('CSV file successfully processed.');
  });
});




app.get('/geth', (req, res) => {
  console.log("GETHISTORY")
  const result = []
  fs.createReadStream('C:\\Users\\hp\\Desktop\\musicc\\server\\input.csv')
  .pipe(csv())
  .on('data', (data) => {
    result.push(data)
    
  })
  .on('end', () => {
    res.send(result)
    console.log('CSV file successfully processed.');
  });
});



app.post('/post', (req, res) => {
  const csvFilePath = 'C:\\Users\\hp\\Desktop\\musicc\\server\\input.csv';

  let rowCount = 0;
  let lastIndex = 0;

  fs.createReadStream(csvFilePath)
    .pipe(csv())
    .on('data', () => {
      rowCount += 1;
      lastIndex = rowCount - 1; // Subtracting 1 to start from 0-based index
    })
    .on('end', () => {
      const name = req.body.name;
      const rating = req.body.rating;
      const newIndex = lastIndex + 1;

      const newData = `\n${newIndex},${name},${rating}`; // Adding the new data with the updated index

      fs.appendFile(csvFilePath, newData, (err) => {
        if (err) {
          console.error('Error appending data to CSV:', err);
          res.status(500).send('Error appending data to CSV');
        } else {
          console.log('Data appended to CSV');
          res.status(200).send('Data appended to CSV');
        }
      });
    });
});






app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

