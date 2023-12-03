const express = require('express');
const { exec, spawn } = require('child_process');

const cors = require('cors');

const app = express();

// Enable CORS for all routes
app.use(cors());
app.use(express.json()); // Add this line before your routes to parse JSON

const PORT = 8080; // You can use any port number you prefer
const pythonScriptPath = 'C:\\Users\\hp\\Desktop\\musicc\\server\\Recommender.py';
const fs = require('fs');
const csv = require('csv-parser');

// app.get('/', (req, res) => {
//   exec(`python C:\\Users\\hp\\Desktop\\musicc\\server\\Recommender.py`, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`Error executing Python script: ${error}`);
//       res.send("Error")
//       return;
//     }
//     console.log(`Python script executed successfully: ${stdout}`);
//     res.send("Success")
//   });
// });



app.get('/', (req, res) => {
    console.log('URL saved successfully!');
    const { spawn } = require('child_process');
    const pythonProcess = spawn('C:/Users/hp/AppData/Local/Programs/Python/Python311/python.exe', ['R.py']);
    let output = ''; // Accumulator for Python output
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString(); // Collect data from stdout
      console.log(`Python stdout: ${data}`);
    });
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
    });
    pythonProcess.on('close', (code) => {
      console.log(`Python process closed with code ${code}`);
      res.json({msg:  "Recommendations Added Refresh the Recommendation Section"}); // Send accumulated output as JSON response
    });
});


app.get('/getr', (req, res) => {
  console.log("GET RECOMMENDATIONS")
  const result = []
  fs.createReadStream('C:\\Users\\hp\\Desktop\\musicc\\server\\output.csv')
  .pipe(csv())
  .on('data', (data) => {
    console.log(data)
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
      const newData = `\n${name}`; // Adding the new data with the updated index
      fs.appendFile(csvFilePath, newData, (err) => {
        if (err) {
          console.error('Error appending data to CSV:', err);
          res.status(500).send({msg :'Error appending data to CSV'});
        } else {
          console.log('Data appended to CSV');
          res.status(200).send({msg: 'Data appended to CSV'});
        }
      });
    });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

