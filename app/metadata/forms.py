# Import key modules required for creating the form and validating the user's input from the wt forms package
from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, SelectMultipleField, RadioField, IntegerField, BooleanField, HiddenField
from wtforms.validators import DataRequired, ValidationError, Optional
from flask_babel import lazy_gettext as _l

# Create a list of all regions to be selected from
regions = [
    ('NAm', _l('North America')), ('LAmC', _l('Latin America and the Caribbean')), ('Eu', _l('Europe')), ('Af', _l('Africa')),
    ('EAs', _l('East Asia')), ('SEAs', _l('South-East Asia')), ('WAs', _l('West Asia')), ('SAs', _l('South Asia')),('Oc', _l('Oceania')), ('Unk', _l('Prefer not to say'))
]

# Create a list of all disability options
disablities = [
    ('prefer_not_to_say', _l('Prefer not to say')), ('none', _l('None')), ('physical', _l('Physical')),
    ('mental_health', _l('Mental Health Condition')), ('learning', _l('Learning Difference (e.g.Dyslexia/Dyspraxia/ADHD)')),
    ('autism', _l("Autism, ASD, or Asperger's Syndrome")), ('developmental', _l('Developmental Disability (e.g. Down Syndrome)')),
]

# Create a form for the user to input their information
class UserInfoForm(FlaskForm):
    age = IntegerField(_l('Age'), validators=[DataRequired()]) # Create an integer field for the user to input their age
    def validate_age(form, field): # Check the user is 18 or older before allowing them to participate
        if field.data < 18:
            raise ValidationError(_l("We're sorry, you must be 18 or older to take part in this study."))

    gender = SelectField(_l('Gender'), choices=[ # Create a dropdown menu for the user to input gender
        ('', _l('Select Gender')), ('male', _l('Male')), ('female', _l('Female')), 
        ('other', _l('Other')), ('prefer_not_to_say', _l('Prefer not to say'))
        ], validators=[DataRequired()])

    sexuality = SelectField(_l('Sexual Orientation'), choices=[ # Create a dropdown menu for the user to input their sexuality
        ('', _l('Select Sexuality')), ('heterosexual', _l('Heterosexual or straight')), ('homosexual', _l('Gay or Lesbian')), ('bisexual', _l('Bisexual')),
        ('other', _l('Other')), ('prefer_not_to_say', _l('Prefer not to say')),
    ], validators=[DataRequired()])

    current_residence_region = SelectField(_l('Where do you currently live?'), choices=[('', _l('Select one region'))] + regions, validators=[DataRequired()])
    upbringing_region = SelectField(_l('Where did you grow up?'), choices=[('', _l('Select one region'))] + regions, validators=[DataRequired()])

    disability = SelectMultipleField(_l('Do you identify as having a disability?'), choices=disablities, validators=[DataRequired()])

    potter = SelectField(_l('Do you have any experience with pottery?'), choices=[ # Create a dropdown menu for the user to input their potter status
        ('', _l('Select Pottery Experience')), ('amateur', _l('Amateur')), ('professional', _l('Professional')),
        ('none', _l('None'))], validators=[DataRequired()])

    consent = BooleanField(_l('I consent to participate in this study'), validators=[DataRequired()]) # Create a checkbox for the user to consent to participate in the study

    # Client context fields are populated by JavaScript on submit.
    browser_language = HiddenField(validators=[Optional()])
    timezone = HiddenField(validators=[Optional()])
    screen_width = HiddenField(validators=[Optional()])
    screen_height = HiddenField(validators=[Optional()])
    device_type = HiddenField(validators=[Optional()])

    def validate_region(form, field):
        if not field.data:
            raise ValidationError(_l("Please select a region."))

    def validate_current_residence_region(form, field):
        form.validate_region(field)
    
    def validate_upbringing_region(form, field):
        form.validate_region(field)

    def validate_disability(form, field):
        if not field.data:
            raise ValidationError(_l("Please select at least one option."))

    def validate_gender(form, field):
        if not field.data:
            raise ValidationError(_l("Please select a gender."))

    def validate_sexuality(form, field):
        if not field.data:
            raise ValidationError(_l("Please select a sexuality."))

    def validate_potter(form, field):
        if not field.data:
            raise ValidationError(_l("Please select pottery experience."))

    submit = SubmitField(_l('Submit')) # Create a submit button for the user to submit their information

