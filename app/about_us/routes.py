
from flask import Blueprint, render_template

# Create a Blueprint for the vase routes to be used in the init file
aboutBP = Blueprint('about_us', __name__)


@aboutBP.route('/participant_information')
def participant_information():
    return render_template('participant_information.html')