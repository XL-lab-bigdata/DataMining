# DataMining

1. 因部分数据集过大无法上传至 GitHub，我们已将其由 .csv 转换为 .parquet 格式，数据内容不变，并同步更新了相关代码。
2. 本书所有代码均为“独立脚本(independent script)”，不做“可引入模块(importable module)”使用，脚本名与书籍章节编号保持一致。若需复用部分代码，请按 pep 8 要求自行更改文件名。

# 书稿配图

<!-- 第一行：三张图片 -->
<p align="center">
  <img src="images/illustrations/1.jpg" width="30%" title="" alt="图片1">
  <img src="images/illustrations/2.jpg" width="30%" title="" alt="图片2">
  <img src="images/illustrations/3.jpg" width="30%" title="" alt="图片3">
</p>

<p align="center">
  <b></b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b></b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b></b>
</p>

<!-- 第二行：三张图片 -->
<p align="center">
  <img src="images/illustrations/4.jpg" width="30%" title="" alt="图片4">
  <img src="images/illustrations/5.jpg" width="30%" title="" alt="图片5">
  <img src="images/illustrations/6.jpg" width="30%" title="" alt="图片6">
</p>

<p align="center">
  <b></b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b></b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b></b>
</p>


# 相关资源


<!DOCTYPE html>
<html>
<head>
    <style>
        /* 整体样式 */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .container {
            max-width: 900px;
            width: 100%;
            background: white;
            border-radius: 15px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            padding: 25px;
            box-sizing: border-box;
        }
        
        h1 {
            color: #3498db;
            text-align: center;
            margin-bottom: 30px;
            font-size: 32px;
        }
        
        /* 轮播器样式 */
        .slider-container {
            position: relative;
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            overflow: hidden;
            border-radius: 10px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        .slider {
            display: flex;
            transition: transform 0.5s ease-in-out;
        }
        
        .slide {
            min-width: 100%;
            height: 450px;
            position: relative;
        }
        
        .slide img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .slide-info {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 20px;
        }
        
        /* 控制按钮样式 */
        .slider-controls {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 25px 0;
        }
        
        .prev-btn, .next-btn {
            background: #3498db;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
            margin: 0 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .prev-btn:hover, .next-btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.25);
        }
        
        .prev-btn:active, .next-btn:active {
            transform: translateY(1px);
        }
        
        /* 指示器样式 */
        .indicators {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-bottom: 15px;
        }
        
        .indicator {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: #ddd;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .indicator.active {
            background-color: #3498db;
            transform: scale(1.2);
        }
        
        /* 图片网格预览 */
        .gallery-container {
            margin-top: 40px;
        }
        
        .grid-title {
            text-align: center;
            color: #2c3e50;
            font-size: 24px;
            margin-bottom: 25px;
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }
        
        .grid-item {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            height: 200px;
        }
        
        .grid-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        
        .grid-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>图片轮播画廊</h1>
        
        <!-- 图片轮播器 -->
        <div class="slider-container">
            <div class="slider" id="slider">
                <div class="slide">
                    <img src="https://picsum.photos/800/450?random=1" alt="自然风景1">
                    <div class="slide-info">
                        <h3>山间日出</h3>
                        <p>清晨的第一缕阳光洒在雪山之上</p>
                    </div>
                </div>
                <div class="slide">
                    <img src="https://picsum.photos/800/450?random=2" alt="自然风景2">
                    <div class="slide-info">
                        <h3>森林迷雾</h3>
                        <p>神秘树林中弥漫着晨雾</p>
                    </div>
                </div>
                <div class="slide">
                    <img src="https://picsum.photos/800/450?random=3" alt="自然风景3">
                    <div class="slide-info">
                        <h3>沙滩日落</h3>
                        <p>夕阳下的金色沙滩</p>
                    </div>
                </div>
                <div class="slide">
                    <img src="https://picsum.photos/800/450?random=4" alt="自然风景4">
                    <div class="slide-info">
                        <h3>高山湖泊</h3>
                        <p>雪山倒影在清澈的湖水中</p>
                    </div>
                </div>
                <div class="slide">
                    <img src="https://picsum.photos/800/450?random=5" alt="自然风景5">
                    <div class="slide-info">
                        <h3>沙漠之星</h3>
                        <p>浩瀚沙漠上空的璀璨星空</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 控制按钮 -->
        <div class="slider-controls">
            <button class="prev-btn" id="prev-btn">❮</button>
            <div class="indicators" id="indicators"></div>
            <button class="next-btn" id="next-btn">❯</button>
        </div>
        
        <!-- 缩略图导航 -->
        <div class="gallery-container">
            <h2 class="grid-title">图片预览导航</h2>
            <div class="image-grid" id="image-grid">
                <!-- 图片将由JavaScript填充 -->
            </div>
        </div>
    </div>

    <script>
        // 轮播器功能
        let slideIndex = 0;
        const slides = document.querySelectorAll('.slide');
        const indicatorsContainer = document.getElementById('indicators');
        const imageGrid = document.getElementById('image-grid');
        
        // 设置轮播器初始状态
        function initSlider() {
            // 创建指示点
            slides.forEach((slide, index) => {
                const indicator = document.createElement('div');
                indicator.classList.add('indicator');
                indicator.addEventListener('click', () => {
                    goToSlide(index);
                });
                indicatorsContainer.appendChild(indicator);
            });
            
            // 创建网格缩略图
            slides.forEach((slide, index) => {
                const gridItem = document.createElement('div');
                gridItem.classList.add('grid-item');
                gridItem.innerHTML = `
                    <img src="${slide.querySelector('img').src}" alt="预览${index + 1}">
                `;
                gridItem.addEventListener('click', () => {
                    goToSlide(index);
                });
                imageGrid.appendChild(gridItem);
            });
            
            updateIndicators();
        }
        
        // 更新指示点状态
        function updateIndicators() {
            const indicators = document.querySelectorAll('.indicator');
            indicators.forEach((indicator, index) => {
                if (index === slideIndex) {
                    indicator.classList.add('active');
                } else {
                    indicator.classList.remove('active');
                }
            });
        }
        
        // 转到指定幻灯片
        function goToSlide(index) {
            if (index >= slides.length) {
                slideIndex = 0;
            } else if (index < 0) {
                slideIndex = slides.length - 1;
            } else {
                slideIndex = index;
            }
            
            document.querySelector('.slider').style.transform = `translateX(-${slideIndex * 100}%)`;
            updateIndicators();
        }
        
        // 下一张幻灯片
        function nextSlide() {
            goToSlide(slideIndex + 1);
        }
        
        // 上一张幻灯片
        function prevSlide() {
            goToSlide(slideIndex - 1);
        }
        
        // 自动轮播
        let slideInterval = setInterval(nextSlide, 5000);
        
        // 鼠标悬停暂停轮播
        document.querySelector('.slider-container').addEventListener('mouseenter', () => {
            clearInterval(slideInterval);
        });
        
        // 鼠标离开恢复轮播
        document.querySelector('.slider-container').addEventListener('mouseleave', () => {
            slideInterval = setInterval(nextSlide, 5000);
        });
        
        // 添加按钮事件
        document.getElementById('prev-btn').addEventListener('click', prevSlide);
        document.getElementById('next-btn').addEventListener('click', nextSlide);
        
        // 初始化轮播器
        document.addEventListener('DOMContentLoaded', initSlider);
    </script>
</body>
</html>

# 购买链接
