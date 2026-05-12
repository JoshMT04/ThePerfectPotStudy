from flask import Blueprint, render_template # Necessary imports for the experiment info page blueprint

experimentBP = Blueprint('experiment', __name__) # Create the experiment info page blueprint to be used in the __init__.py file

@experimentBP.route('/') # Define the url routes for the experiment info page
@experimentBP.route('/experiment')
def experiment():
    return render_template('experiment_info.html') # Render the experiment info page template