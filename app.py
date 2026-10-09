
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Decision Tree model
model = joblib.load("model.pkl")

# Features used by the machine learning model
features = [
    "num_rooms",
    "num_people",
    "housearea",
    "is_ac",
    "is_tv",
    "is_flat",
    "num_children",
    "is_urban"
]


# --------------------------------------------------
# FUNCTION 1: Calculate estimated MSEDCL electricity bill
# --------------------------------------------------

def calculate_bill(units):
    # MSEDCL Residential LT-I tariff estimate for FY 2026-27
    units = max(0, float(units))

    # Energy charges using progressive slabs
    if units <= 100:
        energy_charges = units * 3.96

    elif units <= 300:
        energy_charges = (
            100 * 3.96
            + (units - 100) * 10.80
        )

    elif units <= 500:
        energy_charges = (
            100 * 3.96
            + 200 * 10.80
            + (units - 300) * 15.03
        )

    else:
        energy_charges = (
            100 * 3.96
            + 200 * 10.80
            + 200 * 15.03
            + (units - 500) * 17.53
        )

    # Wheeling charges
    wheeling_charges = units * 1.60

    # Fixed charges for a single-phase residential connection
    fixed_charges = 130.00

    # Additional fixed charge for municipal corporation areas
    # Change to True if this applies to the household.
    is_municipal_corporation = False

    if is_municipal_corporation:
        fixed_charges += 10.00

    # Single-phase meter rent
    meter_rent = 35.00

    # Electricity duty: 16% on energy and fixed charges
    electricity_duty = 0.16 * (
        energy_charges + fixed_charges
    )

    # FAC varies over time. Update this rate when the applicable
    # monthly or quarterly MSEDCL rate is known.
    fac_rate_per_unit = 0.00
    fuel_adjustment_charge = units * fac_rate_per_unit

    # Estimated total bill
    total_bill = (
        energy_charges
        + wheeling_charges
        + fixed_charges
        + meter_rent
        + electricity_duty
        + fuel_adjustment_charge
    )

    return {
        "energy_charges": round(energy_charges, 2),
        "wheeling_charges": round(wheeling_charges, 2),
        "fixed_charges": round(fixed_charges, 2),
        "meter_rent": round(meter_rent, 2),
        "electricity_duty": round(electricity_duty, 2),
        "fuel_adjustment_charge": round(
            fuel_adjustment_charge, 2
        ),
        "total_bill": round(total_bill, 2)
    }


# --------------------------------------------------
# FUNCTION 2: Generate personalized energy suggestions
# --------------------------------------------------

def generate_suggestions(house, predicted_units):

    suggestions = []

    # AC-related suggestion
    if house["is_ac"] == 1:
        suggestions.append({
            "title": "Improve Air Conditioner Efficiency",
            "description": (
                "Set the AC around 24-26°C when comfortable, "
                "clean its filters regularly, and keep doors "
                "and windows closed while cooling."
            )
        })

    # TV-related suggestion
    if house["is_tv"] == 1:
        suggestions.append({
            "title": "Reduce Television Standby Consumption",
            "description": (
                "Switch off the TV when it is not being used "
                "and avoid leaving it on standby for long periods."
            )
        })

    # Household size
    if house["num_people"] >= 4:
        suggestions.append({
            "title": "Manage Shared Appliance Usage",
            "description": (
                "Coordinate the use of washing machines, irons "
                "and other high-power appliances. Run them with "
                "appropriate loads instead of unnecessary cycles."
            )
        })

    # Children in the household
    if house["num_children"] >= 1:
        suggestions.append({
            "title": "Build Energy-Saving Habits",
            "description": (
                "Encourage everyone to switch off lights, fans "
                "and electronic devices when leaving a room."
            )
        })

    # Urban household
    if house["is_urban"] == 1:
        suggestions.append({
            "title": "Choose Energy-Efficient Appliances",
            "description": (
                "When replacing appliances, compare their "
                "electricity consumption and BEE star ratings."
            )
        })

    # Larger house
    if house["housearea"] >= 1000:
        suggestions.append({
            "title": "Control Lighting and Cooling by Room",
            "description": (
                "Switch off lights and fans in unused rooms. "
                "Cool occupied rooms rather than unnecessarily "
                "cooling the entire house."
            )
        })

    # Multiple rooms
    if house["num_rooms"] >= 4:
        suggestions.append({
            "title": "Avoid Unnecessary Room Lighting",
            "description": (
                "Use LED bulbs and natural daylight where "
                "possible, and switch off lights in empty rooms."
            )
        })

    # Suggestions based on predicted consumption
    if predicted_units > 300:
        suggestions.append({
            "title": "Review High Electricity Consumption",
            "description": (
                "Your predicted consumption exceeds 300 units. "
                "Review the use of high-power appliances such "
                "as ACs, water heaters, irons and electric heaters."
            )
        })

    elif predicted_units > 100:
        suggestions.append({
            "title": "Monitor Monthly Electricity Usage",
            "description": (
                "Your predicted consumption exceeds 100 units. "
                "Record monthly meter readings and identify "
                "which appliances contribute most to usage."
            )
        })

    else:
        suggestions.append({
            "title": "Maintain Efficient Energy Habits",
            "description": (
                "Your predicted consumption is at or below "
                "100 units. Continue switching off unused "
                "appliances and using energy-efficient lighting."
            )
        })

    # General suggestions used to fill the list
    general_suggestions = [
        {
            "title": "Use LED Lighting",
            "description": (
                "Replace frequently used conventional bulbs "
                "with suitable LED bulbs."
            )
        },
        {
            "title": "Avoid Unnecessary Standby Power",
            "description": (
                "Switch off devices when not needed. Unplug "
                "equipment only when safe and appropriate."
            )
        },
        {
            "title": "Maintain Household Appliances",
            "description": (
                "Keep appliances clean and maintained according "
                "to their manufacturers' instructions."
            )
        },
        {
            "title": "Use Natural Daylight",
            "description": (
                "Open curtains during daylight when practical "
                "to reduce the need for electric lighting."
            )
        }
    ]

    # Add general suggestions until we have five
    for suggestion in general_suggestions:
        if len(suggestions) >= 5:
            break

        suggestions.append(suggestion)

    # Return no more than five suggestions
    return suggestions[:5]


# --------------------------------------------------
# MAIN WEBSITE ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    bill = None
    suggestions = []
    error = None

    if request.method == "POST":

        try:

            # Read household details from the form
            house = {
                "num_rooms": float(request.form["num_rooms"]),
                "num_people": float(request.form["num_people"]),
                "housearea": float(request.form["housearea"]),
                "is_ac": int(request.form["is_ac"]),
                "is_tv": int(request.form["is_tv"]),
                "is_flat": int(request.form["is_flat"]),
                "num_children": float(request.form["num_children"]),
                "is_urban": int(request.form["is_urban"])
            }

            # Basic input validation
            if any(
                house[key] < 0
                for key in [
                    "num_rooms",
                    "num_people",
                    "housearea",
                    "num_children"
                ]
            ):
                raise ValueError("Household values cannot be negative.")

            if house["num_people"] < 1:
                raise ValueError("At least one person is required.")

            for key in ["is_ac", "is_tv", "is_flat", "is_urban"]:
                if house[key] not in [0, 1]:
                    raise ValueError("Select Yes or No for each option.")

            # Prepare data in the exact order expected by the model
            new_house = pd.DataFrame(
                [house],
                columns=features
            )
            print("\nInput received from website:")
            print(new_house)

            # Predict household electricity consumption
            prediction = float(model.predict(new_house)[0])

            # Avoid a negative prediction
            prediction = max(0.0, prediction)

            # Calculate the estimated electricity bill
            bill = calculate_bill(prediction)

            # Generate personalized suggestions
            suggestions = generate_suggestions(
                house,
                prediction
            )

        except (ValueError, KeyError):
            error = (
                "Please enter valid household details "
                "and select the required options."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        bill=bill,
        suggestions=suggestions,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=False)

