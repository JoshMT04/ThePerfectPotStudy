# Import key modules required for creating the form and validating the user's input from the wt forms package
from flask_wtf import FlaskForm
from wtforms import SubmitField, RadioField, BooleanField, TextAreaField
from wtforms.validators import Optional
from flask_babel import lazy_gettext as _l

class ErrorReportForm(FlaskForm):
    hypothesis = RadioField(_l('I think I know the hypothesis of the experiment.'), choices=[
        ('yes', _l('Yes')), ('no', _l('No'))
    ], validators=[Optional()]) # Create a checkbox for the user to indicate if they think they know the hypothesis of the experiment
    hyp_text = TextAreaField(_l('If so, please describe the hypothesis you think is being tested:'), validators=[Optional()]) # Create a text area for the user to input their hypothesis if they indicated that they think they know it
    report = BooleanField(_l('I had an issue when completing the experiment.')) # Create a checkbox for the user to consent to participate in the study
    issue = RadioField(_l('What issue did you experience?'), choices=[ # Create a dropdown menu for the user to input their potter status
        ('image_not_loading', _l('An image did not load')), ('unable_to_select', _l('I was unable to select a vase')),
        ('other', _l('Other issue'))
    ], validators=[Optional()])
    other_issue = TextAreaField(_l('Please describe the issue you experienced:'), validators=[Optional()])
    submit = SubmitField(_l('Submit Report'))