#import "@preview/typpuccino:0.1.0": latte, frappe, macchiato, mocha
#import "@preview/cades:0.3.0": qr-code

#let poster(
  // The poster's size.
  size: "'36x24' or '48x36''",

  // The poster's title.
  title: "Paper Title",

  // Main message
  main_message: "Main message",

  // A string of author names.
  authors: "Author Names (separated by commas)",

  // Department name.
  departments: "Department Name",

  // University logo.
  univ_logo: "Logo Path",
  footer_logo: "Logo Path",
  main_image: "Logo Path",

  // Footer text.
  // For instance, Name of Conference, Date, Location.
  // or Course Name, Date, Instructor.
  footer_text: "Footer Text",

  // Any URL, like a link to the conference website.
  footer_url: "Footer URL",

  // Email IDs of the authors.
  footer_email_ids: "Email IDs (separated by commas)",

  // Color of the footer.
  footer_color: "Hex Color Code",

  // DEFAULTS
  // ========
  // For 3-column posters, these are generally good defaults.
  // Tested on 36in x 24in, 48in x 36in, and 36in x 48in posters.
  // For 2-column posters, you may need to tweak these values.
  // See ./examples/example_2_column_18_24.typ for an example.

  // Any keywords or index terms that you want to highlight at the beginning.
  keywords: (),

  // Number of columns in the poster.
  num_columns: "3",

  // University logo's scale (in %).
  univ_logo_scale: "100",

  // University logo's column size (in in).
  univ_logo_column_size: "10",

  // Title and authors' column size (in in).
  title_column_size: "20",

  // Poster title's font size (in pt).
  title_font_size: "48",
  main_message_font_size: "48",
  heading_font_size: "48",
  body_font_size: "48",

  // Authors' font size (in pt).
  authors_font_size: "36",

  // Footer's URL and email font size (in pt).
  footer_url_font_size: "30",

  // Footer's text font size (in pt).
  footer_text_font_size: "40",

  // The poster's content.
  body
) = {
  let sizes = size.split("x")
  //let width = float(sizes.at(0)) * 1in
  //let height = float(sizes.at(1)) * 1in
  let width = float(841) * 1mm
  let height = float(1189) * 1mm
  univ_logo_scale = int(univ_logo_scale) * 1%
  title_font_size = int(title_font_size) * 1pt
  main_message_font_size = int(main_message_font_size) * 1pt
  authors_font_size = int(authors_font_size) * 1pt
  num_columns = int(num_columns)
  univ_logo_column_size = int(univ_logo_column_size) * 1in
  title_column_size = int(title_column_size) * 1in
  footer_url_font_size = int(footer_url_font_size) * 1pt
  footer_text_font_size = int(footer_text_font_size) * 1pt
  heading_font_size = int(heading_font_size) * 1pt
  body_font_size = int(body_font_size) * 1pt

  // Set the body font.
  set text(size: body_font_size, fill: latte.text)

  // Configure the page.
  // This poster defaults to 36in x 24in.
  set page(
    width: width,
    height: height,
    fill: latte.base,
    margin: 
      (top: 1in, left: 2in, right: 2in, bottom: 2.8in),
    footer: [
      #set align(center)
      #set text(32pt)
      #move(
        dx: 0in, dy: -1in,
      block(
        fill: latte.surface0,
        width: 100% + 4in,
	height: 4in,
        inset: 0.4in,
        grid(
          columns: (1fr, 1fr, 1fr),
          rows: (2in),
          gutter: 0in,

            grid(columns: (2in, 1fr), qr-code(footer_url, color: latte.text, background: latte.base), align(left+horizon, text(footer_url_font_size, "  " + sym.arrow.l + " Scan for more details!")) ),
            align(center+horizon, text(footer_url_font_size, footer_text)),
            align(right+horizon, image(footer_logo))        
      )))
    ]
  )

  // Configure caption style
  show figure.caption: it => [
  #it.body
  ] 
  show figure.caption: set align(left)

  // Configure equation numbering and spacing.
  set math.equation(numbering: "(1)")
  show math.equation: set block(spacing: 0.65em)

  // Configure lists.
  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)

  // Configure headings.
  //set heading(numbering: "I.A.1.")
  set heading(numbering: "1.")
  show heading: it => locate(loc => {
    // Find out the final number of the heading counter.
    let levels = counter(heading).at(loc)
    let deepest = if levels != () {
      levels.last()
    } else {
      1
    }

    set text(24pt, weight: 400)
    if it.level == 1 [
      // First-level headings are centered smallcaps.
      #set align(left)
      #set text({ 32pt })
      #show: smallcaps
      #v(50pt, weak: true)
      #if it.numbering != none {
        h(7pt, weak: true)
      }
      #text(heading_font_size, it.body, fill: latte.green)
      #v(35.75pt, weak: true)
    ] else if it.level == 2 [
      // Second-level headings are run-ins.
      #set text(style: "italic")
      #v(32pt, weak: true)
      #if it.numbering != none {
        numbering("i.", deepest)
        h(7pt, weak: true)
      }
      #it.body
      #v(10pt, weak: true)
    ] else [
      // Third level headings are run-ins too, but different.
      #if it.level == 3 {
        numbering("1)", deepest)
        [ ]
      }
      _#(it.body):_
    ]
  })


  stack(
  spacing: 0pt,
  move(
  dx: -2in, dy: -1in,
  block(fill: latte.green, spacing: 0%, above: 0%, below: 0%, outset:0%, width: 100% + 4in, inset: 2in, align(center, text(main_message_font_size, main_message, fill: black)))
  ),
  move(
  dx: -2in, dy: -1in,
  block(fill: latte.surface0, spacing: 0%, above: 0%, below: 0%, outset: 0%, width: 100% + 4in, inset: 1in,
  align(left, text(title_font_size, title + "\n") + text(authors_font_size, authors, fill: latte.subtext0) + text(authors_font_size, " - "  + departments, fill: latte.subtext0))
  )
  )
  )

  // Show main image
  align(center,
  image(main_image, width: 93%)
  )

  // Start three column mode and configure paragraph properties.
  show: columns.with(num_columns, gutter: 64pt)
  set par(justify: true, first-line-indent: 0em)
  show par: set block(spacing: 0.65em)

  // Display the keywords.
  if keywords != () [
      #set text(24pt, weight: 400)
      #show "Keywords": smallcaps
      *Keywords* --- #keywords.join(", ")
  ]

  // Display the poster's contents.
  body

}
