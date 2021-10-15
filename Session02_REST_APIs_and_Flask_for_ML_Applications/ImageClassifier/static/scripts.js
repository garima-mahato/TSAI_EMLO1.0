
function scroll_to(clicked_link, nav_height) {
	var element_class = clicked_link.attr('href').replace('#', '.');
	var scroll_to = 0;
	if(element_class != '.top-content') {
		element_class += '-container';
		scroll_to = $(element_class).offset().top - nav_height;
	}
	if($(window).scrollTop() != scroll_to) {
		$('html, body').stop().animate({scrollTop: scroll_to}, 1000);
	}
}

function loadFile(event) {
 
	var image = document.getElementById('input');
	image.src = URL.createObjectURL(event.target.files[0]);
	image.style.paddingLeft = 0;
	
}

function validateForm() {
	var image = document.getElementById('input').files;
	console.log('i'+image);
	if (!image.length) {
	  alert("Please select an image");
	  return false;
	}
}


jQuery(document).ready(function() {
		
	/*
	    Navigation
	*/
	$('a.scroll-link').on('click', function(e) {
		e.preventDefault();
		scroll_to($(this), $('nav').outerHeight());
	});

		// toggle "navbar-no-bg" class
	$('.top-content .text').waypoint(function() {
		$('nav').toggleClass('navbar-no-bg');
	});
	
    /*
        Background slideshow
    */
    $('.top-content').backstretch("static/bg.jpg");
    $('.call-to-action-container').backstretch("static/bg.jpg");
    $('.testimonials-container').backstretch("static/bg.jpg");
    
    $('#top-navbar-1').on('shown.bs.collapse', function(){
    	$('.top-content').backstretch("resize");
    });
    $('#top-navbar-1').on('hidden.bs.collapse', function(){
    	$('.top-content').backstretch("resize");
    });
    
    $('a[data-toggle="tab"]').on('shown.bs.tab', function(){
    	$('.testimonials-container').backstretch("resize");
    });
	
	/* setTimeout(function() {
		$(".flashes").hide('blind', {}, 500)
	}, 5000); */
    
    /*
        Wow
    */
    new WOW().init();
	var obj_load=document.getElementById('settab').value;
	//var tab_load= (obj_load?"#services":"");
	console.log(obj_load);
	
	if(obj_load == 1) {
		link = document.getElementById("try");
		link.click();
		$('a[href="#services"]').tab('show');
		//$('a[href=' + (location.hash ||  tab_load) + ']').tab('show');
	}
});


jQuery(window).load(function() {
	
	/*
		Hidden images
	*/
	$(".testimonial-image img").attr("style", "width: auto !important; height: auto !important;");
	
});

