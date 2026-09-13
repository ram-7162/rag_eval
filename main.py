# Windows paths use \, but Python treats \ as an escape character.
# Resolves:
# (1) Raw string — recommended::
#     r → tells Python to treat \ literally.

# (2) Double backslash
#     \\ → represents one actual backslash.

# (3) Forward slash
#     Works on Windows too.



print("rohit\mohit")
print(r"rohit\mohit")
print("\n")
print(r"\n")