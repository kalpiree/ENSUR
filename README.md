

<h1>ENSUR: Equitable and Statistically Unbiased Recommendation</h1>

<h2>Overview</h2>

<p>Equitable and Statistically Unbiased Recommendation (ENSUR)--> a novel
and reliable framework to dynamically generate prediction sets for users across various groups, which are guaranteed 1) to include
the ground-truth items with user-predefined high confidence/probability (e.g., 90%); 2) to ensure
user fairness across different groups; 3) to have the minimum average set size.</p>

<h2>Project Structure</h2>
<ul>
<li><code>Models/</code> - Directory containing model architectures.</li>
<li><code>ratings_dataset/</code> - Directory containing the dataset files.</li>
<li><code>evaluation.py</code> - Evaluation metrics.</li>
<li><code>lambda_evaluator.py</code> - Evaluation of gamma values on test files.</li>
<li><code>lambda_optimizer.py</code> - Scripts for optimizing lambda parameters.</li>
<li><code>plots_.py</code> - Scripts for generating plots.</li>
<li><code>preprocessing.py</code> - Preprocessing pipeline.</li>
<li><code>result_generator.py</code> - Generates results for final analysis.</li>
<li><code>train.py</code> - Script for training the models.</li>
<li><code>utils.py</code> - Utility functions.</li>
<li><code>run.py</code> - Main script to execute generate scores.</li>
<li><code>requirements.txt</code> - Python dependencies.</li>
</ul>

<h2>Installation</h2>

<p>First, ensure that you have Python installed. Clone this repository and install the necessary dependencies using:</p>

<pre><code>pip install -r requirements.txt
</code></pre>

<h2>How to Run</h2>

<h3>Step 1: Preprocessing</h3>

<p>Run the preprocessing script to prepare the dataset.</p>

<pre><code>python preprocessing.py --input_file '/path/to/your/data/ratings_data.txt' --output_folder './processed_data' --is_implicit --user_top_fraction 0.5 --methods popular_consumption interactions --datasets amazonoffice movielens
</code></pre>

<p><strong>Note:</strong> Replace <code>/path/to/your/data/ratings_data.txt</code> with your actual file path.</p>

<h3>Step 2: Training</h3>

<p>Use the processed data as input to train the models.</p>

<pre><code>python run.py --datasets amazonoffice movielens --models MLP LightGCN --epochs 15 --batch_size 512 --output_folder "results_folder"
</code></pre>

<h3>Step 3: Generate Results</h3>

<p>Take the output from the training step and generate final results.</p>

<pre><code>python result_generator.py --input_folder "results_folder" --output_folder "final_results"
</code></pre>

<h3>Step 4: Visualization</h3>

<p>Finally, generate the plots for fairness and risk metrics.</p>

<pre><code>python result_generator.py --base_folder "path_to_excel_files" --output_folder "path_to_save_plots"
</code></pre>

### Baselines

To evaluate baseline performance, run the scripts located in the `baseline` folder. Detailed instructions are provided within the folder.

---

### New Preprocessed Datasets

The preprocessed datasets include the following grouping strategies:

1. **Interaction Count**:
   - Users were split into two groups, dynamically adjusted so the minimum interaction count in the advantaged group exceeded the maximum in the disadvantaged group.

2. **Age**:
   - Users were grouped into:
     - **Younger Users**: Age ≤ 60 years.
     - **Older Users**: Age > 60 years.

3. **Gender**:
   - Users were grouped into binary categories:
     - **Male**
     - **Female**

4. **Geographic Categorization**:
   - Users were categorized by country:
     - **Developed Countries**: USA, UK, Europe, Japan, etc.
     - **Other Countries**

